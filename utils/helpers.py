import os
import sys
try:
	from openbabel import pybel
except:
	import pybel
import random
import h5py
import pandas as pd
from tqdm import tqdm


from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem
from rdkit.Chem import SDMolSupplier

from utils.sklearn_util import *
from utils.Element_PI import VariancePersistv1

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
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
        from utils.tensorflow_util import nn_basic

        x = x.astype("float32")
        y = y.astype("float32")

        reg = nn_basic(x, y, scale)
    elif algo == "tf_cnn":
        from utils.tensorflow_util import cnn_basic

        x = x.astype("float32")
        y = y.astype("float32")

        reg = cnn_basic(x, y, scale)
    elif algo == "tf_cnn_norm":
        from utils.tensorflow_util import cnn_norm_basic

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
