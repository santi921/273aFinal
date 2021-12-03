import argparse
import pandas as pd

from utils.helpers import calc, qm9
from utils.sklearn_util import *
from sklearn.metrics import r2_score

if __name__ == "__main__":

    ###############IMPORTANT
    ### download qm9 dataset @ https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/
    ### put the dataset in ../data/qm9
    ###############IMPORTANT
    dataset = "qm9"
    target = "homo"
    algo = "tf_cnn"
    desc = "persist"
    ratio = float(1000)

    print("dataset:\t\t" + dataset)
    print("descriptor:\t\t" + desc)
    print("target:\t\t\t" + target)

    mat, target_val = qm9(ratio=ratio, desc=desc, target=target)
    # <---------------------swap these
    # here mat is a list of ndarrays, maybe stack them to (#_of_data, 2500)?
    # mat = np.vstack(mat)
    target_val = np.vstack(target_val)
    scale = np.max(target_val) - np.min(target_val)
    target_val = (target_val - np.min(target_val)) / scale
    reg_model_qm9 = calc(mat, target_val, scale, algo)


    str = "./data/desc/DB3/desc_calc_DB3_persist.h5" #<---------------------swap these
    df = pd.read_hdf(str)

    df_subset = df[['HOMO', "HOMO-1", "diff", "mat"]].sample(n=int(10000), random_state=1)
    HOMO = df_subset["HOMO"].to_numpy()
    mat = df_subset["mat"].to_numpy()
    HOMO = np.vstack(HOMO)

    # scaling
    scale = np.max(HOMO) - np.min(HOMO)
    target_val = (HOMO - np.min(HOMO)) / scale


    reg_model_qm9
    score = str(r2_score(reg_model_qm9.predict(scale), target_val))
