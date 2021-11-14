import os
import sys
import time
import pandas as pd
import numpy as np
import pybel

from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem
from rdkit.Chem import SDMolSupplier

from sklearn_util import *


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
        from tensorflow_util import nn_basic

        x = x.astype("float32")
        y = y.astype("float32")

        reg = nn_basic(x, y, scale)
    elif algo == "tf_cnn":
        from tensorflow_util import cnn_basic

        x = x.astype("float32")
        y = y.astype("float32")

        reg = cnn_basic(x, y, scale)
    elif algo == "tf_cnn_norm":
        from tensorflow_util import cnn_norm_basic

        x = x.astype("float32")
        y = y.astype("float32")

        reg = cnn_norm_basic(x, y, scale)
    elif algo == "resnet":
        from tensorflow_util import resnet34

        x = x.astype("float32")
        y = y.astype("float32")

        reg = resnet34(x, y, scale)
    else:
        print("stochastic gradient descent selected")
        reg = sgd(x, y, scale)
    return reg


def merge_dir_and_data(dir="DB3"):
    # all files in the directory
    ls_dir = "ls " + str(dir) + " | sort"
    dir_fl_names = os.popen(ls_dir).read()
    dir_fl_names = str(dir_fl_names).split()
    dir_fl_names.sort()

    # all energies in the database
    list_to_sort = []
    remove_ind = []
    added_seg = "BQ"
    with open("../data/DATA_DB3") as fp:
        line = fp.readline()
        while line:
            if line.split()[0] == "----":
                added_seg = line.split()[1]
            else:
                list_to_sort.append(added_seg + "_" + line[0:-2])
            line = fp.readline()

    list_to_sort.sort()
    only_names = [i[0:-2].split(":")[0] for i in list_to_sort]

    # find the values which are in the dir and database
    files_relevant = []
    for i, file_name in enumerate(dir_fl_names):
        try:
            ind_find = only_names.index(file_name[0:-4])
            sys.stdout.write("\r %s /" % i + str(len(dir_fl_names)))
            sys.stdout.flush()
            files_relevant.append(file_name)
        except:
            ind_empty = dir_fl_names.index(file_name)
            remove_ind.append(ind_empty)
            pass

    remove_ind.reverse()
    [dir_fl_names.pop(i) for i in remove_ind]
    remove_ind_2 = []

    for ind, files in enumerate(only_names):
        try:
            try:
                ind_find = dir_fl_names.index(files + ".sdf")
            except:
                ind_find = dir_fl_names.index(files + ".xyz")
            sys.stdout.write("\r %s /" % ind + str(len(only_names)))
            sys.stdout.flush()
        except:
            ind_empty = only_names.index(files)
            remove_ind_2.append(ind_empty)
            pass

    remove_ind_2.reverse()
    [only_names.pop(i) for i in remove_ind_2]
    [list_to_sort.pop(i) for i in remove_ind_2]
    # go back and remove files in directory that we don't have energies
    return dir_fl_names, list_to_sort


def morgan(bit_length=256, dir="../data/sdf/DB3/", bit=True):
    morgan = []
    morgan_bit = []
    names = []
    homo = []
    homo1 = []
    diff = []

    dir_fl_names, list_to_sort = merge_dir_and_data(dir=dir)
    print("files to process: " + str(len(dir_fl_names)))
    # ---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            suppl = SDMolSupplier(dir + item)

            if bit == True:
                try:
                    fp_bit = AllChem.GetMorganFingerprintAsBitVect(
                        suppl[0], int(2), nBits=int(bit_length)
                    )
                    morgan.append(fp_bit)
                except:
                    print("error")
                    pass
            else:
                try:
                    fp = AllChem.GetMorganFingerprint(suppl[0], int(2))
                    morgan.append(fp)
                except:
                    print("error")
                    pass

            if item[0:-4] == list_to_sort[tmp].split(":")[0]:
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if item[0:-4] == list_to_sort[tmp + 1].split(":")[0]:
                        morgan.append(fp)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp + 1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp + 1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    print(list_to_sort[tmp].split(":")[0], item[0:-4])
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    morgan = np.array(morgan)
    return names, morgan, homo, homo1, diff


def rdk(dir="../data/sdf/DB/"):
    rdk = []
    names = []
    homo = []
    homo1 = []
    diff = []

    dir_fl_names, list_to_sort = merge_dir_and_data(dir=dir)
    # ---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            suppl = SDMolSupplier(dir + item)
            fp_rdk = AllChem.RDKFingerprint(suppl[0], maxPath=2)

            if item[0:-4] == list_to_sort[tmp].split(":")[0]:
                rdk.append(fp_rdk)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if item[0:-4] == list_to_sort[tmp + 1].split(":")[0]:
                        rdk.append(fp_rdk)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp + 1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp + 1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    print(list_to_sort[tmp].split(":")[0], item[0:-4])
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    rdk = np.array(rdk)
    return names, rdk, homo, homo1, diff


def aval(dir="../data/sdf/DB/", bit_length=256):
    aval = []
    names = []
    homo = []
    homo1 = []
    diff = []
    dir_fl_names, list_to_sort = merge_dir_and_data(dir=dir)
    # ---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            suppl = SDMolSupplier(dir + item)
            fp_aval = pyAvalonTools.GetAvalonFP(suppl[0], bit_length)

            if item[0:-4] == list_to_sort[tmp].split(":")[0]:
                aval.append(fp_aval)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if item[0:-4] == list_to_sort[tmp + 1].split(":")[0]:
                        aval.append(fp_aval)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp + 1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp + 1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    print(list_to_sort[tmp].split(":")[0], item[0:-4])
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    aval = np.array(layer)
    return names, aval, homo, homo1, diff


def layer(dir="../data/sdf/DB/"):
    layer = []
    names = []
    homo = []
    homo1 = []
    diff = []
    dir_fl_names, list_to_sort = merge_dir_and_data(dir=dir)
    # ---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            suppl = SDMolSupplier(dir + item)
            fp_layer = AllChem.LayeredFingerprint(suppl[0])

            if item[0:-4] == list_to_sort[tmp].split(":")[0]:
                layer.append(fp_layer)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if item[0:-4] == list_to_sort[tmp + 1].split(":")[0]:
                        layer.append(fp_layer)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp + 1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp + 1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    print(list_to_sort[tmp].split(":")[0], item[0:-4])
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    layer = np.array(layer)
    return names, layer, homo, homo1, diff


# this script converts xyz files to rdkit/openbabel-readable sdf
# Input: not implemented here but a directory with xyz files
# Input: directory of xyz files
# Output: None, saves SDF type files to and sdf folder for later
def xyz_to_sdf(dir="../data/xyz/DB/"):
    dir_str = "ls " + str(dir) + " | sort "
    temp = os.popen(dir_str).read()
    temp = str(temp).split()

    for j, i in enumerate(temp):
        try:
            i = (
                i.replace("(", "\(")
                .replace(")", "\)")
                .replace("[", "\[")
                .replace("]", "\]")
            )

            file_str = (
                "python ./xyz2mol/xyz2mol.py "
                + dir
                + i
                + " -o sdf > ../data/sdf/"
                + i[0:-4]
                + ".sdf"
            )
            os.system(file_str)
            sys.stdout.write("\r %s / " % j + str(len(temp)))
            sys.stdout.flush()

        except:
            print("not working")


# Input: directory of xyz files
# Output: returns a list of smiles strings
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


def qm9(ratio=0.01, desc="morg", target="HOMO"):
    pixelsx = 50
    pixelsy = 50
    spread = 0.28
    Max = 2.5

    if (target == "HOMO"):
        target_index = 7
    if (target == "LUMO"):
        target_index = 8
    if (target == "gap"):
        target_index = 9
    if (target == "zeropoint"):
        target_index = 11
    if (target == "U0"):
        target_index = 12
    if (target == "G"):
        target_index = 15

    files = os.listdir("../data/qm9/xyz/")
    if (ratio < 1):
        files = random.sample(files, int(len(files) * ratio))

    x_arr = []
    y_arr = []

    for ind, file in enumerate(tqdm(files)):
        file_full = "../data/qm9/xyz/" + file

        if (desc == "persist"):
            try:

                temp_persist = VariancePersistv1(
                    file_full,
                    pixelx=pixelsx, pixely=pixelsy,
                    myspread=spread,
                    myspecs={"maxBD": Max, "minBD": -.10},
                    showplot=False)

                file_obj = open(file_full)
                _ = file_obj.readline()
                target_str = file_obj.readline()
                target = float(target_str.split()[target_index])

                x_arr.append(temp_persist)
                y_arr.append(target)

            except:
                print(file)

        elif (desc == "morg"):
            mol_temp = next(pybel.readfile("xyz", file_full))
            fingerprint_temp = mol_temp.calcfp("fp2").bits
            fingerprint_vect = np.zeros(1022)
            for i in fingerprint_temp: fingerprint_vect[i] = 1
            file_obj = open(file_full)
            _ = file_obj.readline()
            target_str = file_obj.readline()
            target = float(target_str.split()[target_index])
            x_arr.append(fingerprint_vect)
            y_arr.append(target)

    return x_arr, y_arr