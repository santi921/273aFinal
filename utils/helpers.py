import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense,
    MaxPooling2D,
    Conv2D,
    Dropout,
    Flatten,
    BatchNormalization,
    Activation,
    GlobalAvgPool2D,
)
from tensorflow.keras.models import Sequential

import os
import sys
try:
	from openbabel import pybel
except:
	import pybel
import h5py
from tqdm import tqdm
from utils.sklearn_util import *
from utils.Element_PI import VariancePersistv1

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

##################################################################
# Description of function:
# A handler that take x, y matrices, along with scaling and trains algos
# returns: the trained regressor
##################################################################

def calc(x, y, scale, algo="sgd"):

    print("........starting single algo evaluation........")
    if algo == "nn":
        print("nn reg selected")
        reg = sk_nn(x, y, scale)
    elif algo == "rf":
        print("random forest selected")
        reg = random_forest(x, y, scale)
    elif algo == "extra":
        print("extra trees selected")
        reg = extra_trees(x, y, scale)
    elif algo == "grad":
        print("grad algo selected")
        reg = gradient_boost_reg(x, y, scale)
    elif algo == "svr":
        print("svr algo selected")
        reg = svr(x, y, scale)
    elif algo == "bayes":
        print("bayes regression selected")
        reg = bayesian(x, y, scale)
    elif algo == "kernel":
        print("kernel regression selected")
        reg = kernel(x, y, scale)
    elif algo == "gaussian":
        print("gaussian algo selected")
        reg = gaussian(x, y, scale)

    elif algo == "tf_nn":

        x = x.astype("float32")
        y = y.astype("float32")
        reg = nn_basic(x, y, scale)

    elif algo == "tf_cnn":

        x = x.astype("float32")
        y = y.astype("float32")
        reg = cnn_basic(x, y, scale)

    elif algo == "tf_cnn_norm":

        x = x.astype("float32")
        y = y.astype("float32")
        reg = cnn_norm_basic(x, y, scale)

    elif algo == "resnet":
        print("import")
        from utils.tensorflow_util import resnet34
        print("TRYING TO USE RESNET")
        x = x.astype("float32")
        y = y.astype("float32")

        reg = resnet34(x, y, scale)
    else:
        print("stochastic gradient descent selected")
        reg = sgd(x, y, scale)
    return reg


##################################################################
# Description of function:
# converts from xyz files to smiles string representations
# Input: directory of xyz files
# Output: returns a list of smiles strings
##################################################################

def xyz_to_smiles(dir="../data/xyz/DB2/"):
    dir_str = "ls " + str(dir) + " | sort -d "
    temp = os.popen(dir_str).read()
    temp = str(temp).split()
    ret_list = []
    names = []
    for j, i in enumerate(temp):
        try:
            mol = next(pybel.readfile("xyz", dir + i))
            smi = mol.write(format="smi")
            ret_list.append(smi.split()[0].strip())
            names.append(i)
            sys.stdout.write("\r %s / " % j + str(len(temp)))
            sys.stdout.flush()

        except:
            pass
    # print(ret_list[0:4])
    return names, ret_list


# splits a single large smi file into many smaller ones
def smi_split(file=""):
    for i, mol in enumerate(pybel.readfile("smi", "zz.smi")):
        temp = str(i)
        mol.write("smi", "%s.smi" % temp)


# converts a log files of smiles strings to a pandas db of xyz
def smiles_to_xyz(dir="../data/smiles/ZZ/"):
    dir_str = "ls " + str(dir) + " | sort"
    temp = os.popen(dir_str).read()
    temp = str(temp).split()
    t = []
    for i in temp:
        t1 = time.time()
        print("Current file: " + i)
        mol = next(pybel.readfile("smi", dir + i))
        mol.make3D(forcefield="mmff94", steps=10)
        mol.localopt()
        t2 = time.time()
        print("Smi Optimization Complete in " + str(t2 - t1) + "s")
        mol.write("xyz", "%s.xyz" % i)
        t.append(t2 - t1)
        # mol.write("xyz", "%s.xyz" % temp)
    time_array = np.array(t)
    print("time average for computation: " + np.mean(time_array))


##################################################################
# Description of function:
# handles the qm9 dataset, make sure you downloaded it!!!
# Input: ratio is % of the qm9 dataset to pull, desc is descriptor to us, target has some options
# Returns: x, y matrices for training
##################################################################

def qm9(ratio=0.01, desc="morg", target="HOMO"):
    # constants for persistence images
    x_arr = []
    y_arr = []
    pixelsx = 50
    pixelsy = 50
    spread = 0.28
    Max = 2.5

    # potential target variables
    if target == "homo":
        target_index = 7
    elif target == "lumo":
        target_index = 8
    elif target == "diff":
        target_index = 9
    elif target == "zeropoint":
        target_index = 11
    elif target == "U0":
        target_index = 12
    elif target == "G":
        target_index = 15
    else:
        print("invalid target specified")

    # try to load precomputed persistent image
    precomputed_filepath = "./data/qm9/precomputed_qm9.h5" 
    try:
        with h5py.File(precomputed_filepath, 'r') as f:
            print("key list", f.keys())
            # try to load persistent images or morgan fingerprints
            x_arr = f[desc][:]
            y_arr = f["target"][:]
            print(f"desc={desc}, target={target} loaded from {precomputed_filepath}")
        # sample total * ratio rows
        sample_rows = np.random.choice(x_arr.shape[0], size=int(x_arr.shape[0]*ratio), replace=False)
        return x_arr[sample_rows, ...], y_arr[sample_rows, target_index]
    except:
        x_arr, y_arr = [], []
        print("Cannot load from precomputed files, computing and saving to hdf5 files...")

    # if no precomputed exists, precompute desc/target and save to hdf5 file
    files = os.listdir("./data/qm9/xyz/")

    # if ratio < 1:
    #     files = random.sample(files, int(len(files) * ratio))

    for ind, file in enumerate(tqdm(files)):
        file_full = "./data/qm9/xyz/" + file
        if desc == "persist":
            try:
                temp_persist = VariancePersistv1(
                    file_full,
                    pixelx=pixelsx,
                    pixely=pixelsy,
                    myspread=spread,
                    myspecs={"maxBD": Max, "minBD": -0.10},
                    showplot=False,
                )

                file_obj = open(file_full)
                _ = file_obj.readline()
                target_str = file_obj.readline()
                # instead of reading one target, storing all targets into array
                # target = float(target_str.split()[target_index])
                targets = [0] + [float(t) for t in target_str.split()[1:]]

                x_arr.append(temp_persist)
                y_arr.append(targets)

            except:
                print(file)

        elif desc == "morg":
            print("calculating morgan fingerprints")
            mol_temp = next(pybel.readfile("xyz", file_full))

            fingerprint_temp = mol_temp.calcfp("fp2").bits
            fingerprint_vect = np.zeros(1022) # length of hashing array

            for i in fingerprint_temp:
                fingerprint_vect[i] = 1
            file_obj = open(file_full)
            _ = file_obj.readline()
            target_str = file_obj.readline()
            # instead of reading one target, storing all targets into array
            # target = float(target_str.split()[target_index])
            target = targets = [0] + [float(t) for t in target_str.split()[1:]]
            targets = [0] + [float(t) for t in target_str.split()[1:]]

            x_arr.append(fingerprint_vect)
            y_arr.append(targets)
            

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    # save to hdf5 file
    with h5py.File(precomputed_filepath, 'w') as f:
        f.create_dataset(name=desc, data=x_arr)
        f.create_dataset(name="target", data=y_arr)
        print(f"Saved computed \"{desc}\" and \"all targets\" to \"{precomputed_filepath}\".")

    # sample total * ratio rows
    sample_rows = np.random.choice(x_arr.shape[0], size=int(x_arr.shape[0]*ratio), replace=False)

    return x_arr[sample_rows, ...], y_arr[sample_rows, target_index]


def nn_basic(x, y, scale, iter=50):
    print("setting gpu options")
    try:
        x.shape
        x = x.tolist()
    except:
        pass

    try:
        x = tf.convert_to_tensor(x.tolist())
        y = tf.convert_to_tensor(y.tolist())
        input_dim = np.shape(x[0])

    except:
        input_dim = len(x[0])

    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )
    print("Input vector size: " + str(input_dim))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))

    model.add(tf.keras.layers.Dense(1024, activation="relu"))
    model.add(Dropout(0.25))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, activation="relu"))

    model.add(tf.keras.layers.Dense(1, activation="linear"))
    model.summary()

    # mae = tf.keras.losses.MAE()
    # rmse = tf.keras.losses.RMSE()
    model.compile(optimizer="adam", loss="MSE", metrics=["MeanSquaredError", "MAE"])
    early_stop = EarlyStopping(monitor="loss", verbose=1, patience=10)
    early_stop = EarlyStopping(monitor="loss", verbose=1, patience=10)
    history = model.fit(
        x_train, y_train, epochs=iter, validation_split=0.15, callbacks=[early_stop]
    )

    ret = model.evaluate(x_test, y_test, verbose=1)
    print(history.history.keys())
    plt.plot(history.history["loss"][2:-1], label="Training Loss")
    plt.plot(history.history["val_loss"][2:-1], label="Validation Loss")
    plt.legend()

    score = str(mean_squared_error(model.predict(x_test), y_test))
    print("MSE score:   " + str(score))
    score = str(mean_absolute_error(model.predict(x_test), y_test))
    print("MAE score:   " + str(score))
    score = str(r2_score(model.predict(x_test), y_test))
    print("r2 score:   " + str(score))
    score_mae = mean_absolute_error(model.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return model


def cnn_basic(x, y, scale, iter=50):
    print("setting gpu options")

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    try:
        x.shape
        x = x.tolist()
    except:
        pass

    try:
        x = tf.convert_to_tensor(x.tolist())
        y = tf.convert_to_tensor(y.tolist())
        input_dim = np.shape(x[0])

    except:
        input_dim = len(x[0])

    x = np.array(x)
    y = np.array(y)

    dim_persist = int(np.shape(x)[1] ** 0.5)
    x = x.reshape((np.shape(x)[0], dim_persist, dim_persist))
    x = np.expand_dims(x, -1)
    print(np.shape(x))

    samples = int(np.shape(x)[0])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )
    print("Input vector size: " + str(input_dim))
    model = Sequential()

    model.add(
        Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            input_shape=(dim_persist, dim_persist, 1),
            strides=1,
            data_format="channels_last",
        )
    )
    model.add(MaxPooling2D(pool_size=2))

    model.add(
        Conv2D(
            filters=32,
            kernel_size=2,
            activation="relu",
            strides=1,
            data_format="channels_last",
        )
    )
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="linear"))

    model.summary()
    # mae = tf.keras.losses.MAE()
    # rmse = tf.keras.losses.RMSE()
    log_dir = "./logs/training/"
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # model.compile(optimizer='adam', loss=mse, metrics=[keras.metrics.mae])
    model.compile(optimizer="adam", loss="MSE", metrics=["MeanSquaredError", "MAE"])

    # tensorboard_cbk = TensorBoard(log_dir=log_dir)
    history = model.fit(x_train, y_train, epochs=iter, validation_split=0.15)
    ret = model.evaluate(x_test, y_test, verbose=2)

    print(history.history.keys())
    plt.plot(history.history["loss"][2:-1], label="Training Loss")
    plt.plot(history.history["val_loss"][2:-1], label="Validation Loss")
    plt.legend()

    score = str(mean_squared_error(model.predict(x_test), y_test))
    print("MSE score:   " + str(score))

    score = str(mean_absolute_error(model.predict(x_test), y_test))
    print("MAE score:   " + str(score))

    score = str(r2_score(model.predict(x_test), y_test))
    print("r2 score:   " + str(score))

    score_mae = mean_absolute_error(model.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return model


def cnn_norm_basic(x, y, scale, iter=200):
    print("setting gpu options")

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    try:
        x.shape
        x = x.tolist()
    except:
        pass

    try:
        x = tf.convert_to_tensor(x.tolist())
        y = tf.convert_to_tensor(y.tolist())
        input_dim = np.shape(x[0])

    except:
        input_dim = len(x[0])

    x = np.array(x)
    y = np.array(y)

    dim_persist = int(np.shape(x)[1] ** 0.5)
    x = x.reshape((np.shape(x)[0], dim_persist, dim_persist))
    x = np.expand_dims(x, -1)
    print(np.shape(x))

    samples = int(np.shape(x)[0])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )
    print("Input vector size: " + str(input_dim))
    model = Sequential()

    model.add(
        Conv2D(
            filters=64,
            kernel_size=3,
            input_shape=(dim_persist, dim_persist, 1),
            strides=1,
            data_format="channels_last",
        )
    )
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2))

    # model.add(Conv2D(filters=64, kernel_size=3,
    #                 strides=1, data_format="channels_last"))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=2))
    # model.add(Activation("relu"))

    model.add(Conv2D(filters=64, kernel_size=3, strides=1, data_format="channels_last"))
    model.add(BatchNormalization())
    # model.add(local)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation="linear"))

    model.summary()
    # mae = tf.keras.losses.MAE()
    # rmse = tf.keras.losses.RMSE()
    log_dir = "./logs/training/"
    model.compile(optimizer="adam", loss="MSE", metrics=["MeanSquaredError", "MAE"])

    history = model.fit(x_train, y_train, epochs=iter, validation_split=0.15)
    ret = model.evaluate(x_test, y_test, verbose=2)
    print(history.history.keys())
    plt.plot(history.history["loss"][2:-1], label="Training Loss")
    plt.plot(history.history["val_loss"][2:-1], label="Validation Loss")
    plt.legend()

    score = str(mean_squared_error(model.predict(x_test), y_test))
    print("MSE score:   " + str(score))

    score = str(mean_absolute_error(model.predict(x_test), y_test))
    print("MAE score:   " + str(score))

    score = str(r2_score(model.predict(x_test), y_test))
    print("r2 score:   " + str(score))

    score_mae = mean_absolute_error(model.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return model



