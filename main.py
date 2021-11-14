import argparse, math, joblib, uuid

import numpy as np
import pandas as pd
from utils.helpers import calc, qm9

from sklearn import preprocessing
from utils.sklearn_util import *

if __name__ == "__main__":

    ###############IMPORTANT
    ### download qm9 dataset @ https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/
    ### put the dataset in ../data/qm9
    ###############IMPORTANT

    parser = argparse.ArgumentParser(description="select dataset, algorithm")

    parser.add_argument(
        "-dataset",
        action="store",
        dest="dataset",
        default="morgan",
        help="select dataset: [morgan, persist, qm9]",
    )

    parser.add_argument(
        "--algo",
        action="store",
        dest="algo",
        default="DB",
        help="options: [svr_rbf, svr_poly, svr_lin, grad, rf, sgd, bayes, kernel, gaussian, nn,\
    resnet, tf_nn, tf_cnn, tf_cnn_norm]",
    )
    parser.add_argument("--diff", dest="diff", action="store_true")
    parser.add_argument("--homo", dest="homo", action="store_true")
    parser.add_argument("--homo1", dest="homo1", action="store_true")

    results = parser.parse_args()
    print("parser parsed")

    des = results.desc
    dataset = results.dataset
    results = parser.parse_args()
    algo = results.algo
    diff_tf = results.diff
    homo_tf = results.homo
    homo1_tf = results.homo1

    print("pulling dataset: " + dataset)

    if homo1_tf == False and homo_tf == False:
        diff_tf = True

    if dataset == "qm9":
        mat, target_val = qm9()
        scale = np.max(target_val) - np.min(target_val)
        target_val = (target_val - np.min(target_val)) / scale

    else:
        if dataset == "morgan":
            str = "../data/desc/desc_calc_morg.h5"
            str2 = "../data/desc/desc_calc_morg.pkl"
        elif dataset == "persist":
            str = "../data/desc/desc_calc_persist.h5"
            str2 = "../data/desc/desc_calc_persist.pkl"

        try:
            print(str)
            df = pd.read_pickle(str)
            pkl = 1
        except:
            print(str2)
            df = pd.read_hdf(str2)
            pkl = 0

        print(len(df))
        print(df.head())
        HOMO = df["HOMO"].to_numpy()
        HOMO_1 = df["HOMO-1"].to_numpy()
        diff = df["diff"].to_numpy()
        mat = df["mat"].to_numpy()

        try:
            mat = preprocessing.scale(np.array(mat))
        except:
            mat = list(mat)
            mat = preprocessing.scale(np.array(mat))

        print("Using " + des + " as the descriptor")
        print("Matrix Dimensions: {0}".format(np.shape(mat)))

        # finish optimization
        if homo_tf == True:
            des = des + "_homo"
            print(".........................HOMO..................")
            scale = np.max(HOMO) - np.min(HOMO)
            target_val = (HOMO - np.min(HOMO)) / scale

        elif homo1_tf == True:
            des = des + "_homo_1"
            print(".........................HOMO1..................")
            scale = np.max(HOMO_1) - np.min(HOMO_1)
            target_val = (HOMO_1 - np.min(HOMO_1)) / scale

        else:
            des = des + "_diff"
            print(".........................diff..................")
            scale = np.max(diff) - np.min(diff)
            target_val = (diff - np.min(diff)) / scale

        reg_model = calc(mat, target_val, scale, algo)
