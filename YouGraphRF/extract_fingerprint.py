import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.EState import Fingerprinter
from ogb.graphproppred import GraphPropPredDataset

def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 3, useChirality=True))

def getestatefingerprint(mol):
    x = np.concatenate(Fingerprinter.FingerprintMol(mol))
    return list(x)

def getmaccsfingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

def main(dataset_path, is_train=True):
    df_smi = pd.read_csv(dataset_path)
    smiles = df_smi["clean"]

    mgf_feat_list = []
    maccs_feat_list = []
    estate_feat_list = []
    for ii in tqdm(range(len(smiles))):
        rdkit_mol = AllChem.MolFromSmiles(smiles.iloc[ii])

        mgf = getmorganfingerprint(rdkit_mol)
        mgf_feat_list.append(mgf)

        maccs = getmaccsfingerprint(rdkit_mol)
        maccs_feat_list.append(maccs)

        estate_feat_list.append(getestatefingerprint(rdkit_mol))

    mgf_feat = np.array(mgf_feat_list, dtype="int64")
    maccs_feat = np.array(maccs_feat_list, dtype="int64")
    estate_feat = np.array(estate_feat_list)
    print("morgan feature shape: ", mgf_feat.shape)
    print("maccs feature shape: ", maccs_feat.shape)

    save_path = f"./".replace("-", "_")
    print("saving feature in %s" % save_path)

    if is_train:
        np.save(os.path.join(save_path, "mgf_feat.npy"), mgf_feat)
        np.save(os.path.join(save_path, "maccs_feat.npy"), maccs_feat)
        np.save(os.path.join(save_path, "estate_feat.npy"), estate_feat)
    else:
        np.save(os.path.join(save_path, "mgf_feat_test.npy"), mgf_feat)
        np.save(os.path.join(save_path, "maccs_feat_test.npy"), maccs_feat)
        np.save(os.path.join(save_path, "estate_feat_test.npy"), estate_feat)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--train_data", type=str, default="../../../Downloads/Telegram Desktop/clean_train_with_FP2.csv")
    parser.add_argument("--test_data", type=str, default="../../../Downloads/Telegram Desktop/clean_test_with_FP2.csv")
    args = parser.parse_args()
    main(args.train_data)
    main(args.test_data, False)
