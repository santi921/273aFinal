import argparse
import pandas as pd

from utils.helpers import calc, qm9
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
        default="quinone",
        help="select dataset: [quinone, qm9]",
    )

    parser.add_argument(
        "-algo",
        action="store",
        dest="algo",
        default="sgd",
        help="options: [svr_rbf, svr_poly, svr_lin, grad, rf, sgd, bayes, kernel, gaussian, nn,\
    resnet, tf_nn, tf_cnn, tf_cnn_norm]"
    )

    parser.add_argument(
        "-target",
        action="store",
        dest="target",
        default="homo",
        help="options: [quinone: homo, homo1, diff / qm9: homo, lumo, diff, zeropoint, U0, G]"
    )

    parser.add_argument(
        "-desc",
        action="store",
        dest="desc",
        default="morg",
        help="options: [morg, persist]",
    )

    parser.add_argument(
        "-ratio",
        action="store",
        dest="ratio",
        default="0.01",
        help="how much of the full dataset to work on"
    )
    
    results = parser.parse_args()
    dataset = results.dataset
    target = results.target
    algo = results.algo
    desc = results.desc
    ratio = float(results.ratio)

    print("dataset:\t\t" + dataset)
    print("descriptor:\t\t" + desc)
    print("target:\t\t\t" + target)

    if dataset == "qm9":
        # todo: precompute full persistent and save??
        mat, target_val = qm9(ratio = ratio, desc = desc, target = target)
        scale = np.max(target_val) - np.min(target_val)
        target_val = (target_val - np.min(target_val)) / scale
        reg_model = calc(mat, target_val, scale, algo)

    else:
        if desc == "morg":
            str = "./data/desc/DB3/desc_calc_DB3_morg.h5"
            str2 = "./data/desc/DB3/desc_calc_DB3_morg.pkl"

        elif desc == "persist":
            str = "./data/desc/DB3/desc_calc_DB3_persist.h5"
            str2 = "./data/desc/DB3/desc_calc_DB3_persist.pkl"

        else:
            print("INVALID DESCRIPTOR SELECTED")

        try:
            df = pd.read_pickle(str2)
            print(str2)
        except:
            df = pd.read_hdf(str)
            print(str)

        # subsetting
        if ratio < 1:
            df_subset = df[['HOMO', "HOMO-1", "diff", "mat"]].sample(n=int(ratio * df.shape[0]), random_state=1)
            HOMO = df_subset["HOMO"].to_numpy()
            HOMO_1 = df_subset["HOMO-1"].to_numpy()
            diff = df_subset["diff"].to_numpy()
            mat = df_subset["mat"].to_numpy()

        else:
            HOMO = df["HOMO"].to_numpy()
            HOMO_1 = df["HOMO-1"].to_numpy()
            diff = df["diff"].to_numpy()
            mat = df["mat"].to_numpy()

        # scaling
        if target == 'homo':
            scale = np.max(HOMO) - np.min(HOMO)
            target_val = (HOMO - np.min(HOMO)) / scale

        elif target == 'homo1':
            scale = np.max(HOMO_1) - np.min(HOMO_1)
            target_val = (HOMO_1 - np.min(HOMO_1)) / scale

        else:
            scale = np.max(diff) - np.min(diff)
            target_val = (diff - np.min(diff)) / scale

        reg_model = calc(mat, target_val, scale, algo)
