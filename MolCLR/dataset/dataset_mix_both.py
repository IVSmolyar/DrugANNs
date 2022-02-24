import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy
import networkx as nx

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.EState import Fingerprinter


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def remove_subgraph(Graph, center, percent=0.2):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes)*percent))
    removed = []
    temp = [center]
    
    while len(removed) < num:
        neighbors = []
        if len(temp) < 1:
            break

        for n in temp:
            if not G.has_node(n):
                continue
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])      
        for n in temp:
            if len(removed) < num:
                if G.has_node(n):
                    G.remove_node(n)
                removed.append(n)
            else:
                break

        temp = list(set(neighbors))
    return G, removed


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 231231312213123:
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels


class MoleculeDataset(Dataset):
    def __init__(self, data_path, target, task):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = "classification"

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        x = np.array(list(Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2)))
        y = np.concatenate(Fingerprinter.FingerprintMol(mol))
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        fp = np.array([int(b) for b in fp.ToBitString()])
        # print(x.shape)
        # print(y.shape)
        # print(fp.shape)
        arr = torch.FloatTensor(np.concatenate([x, y, fp]))
        # print(arr.shape)

        #########################
        # Get the molecule info #
        #########################
        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        ####################
        # Subgraph Masking #
        ####################

        # Construct the original molecular graph from edges (bonds)
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        molGraph = nx.Graph(edges)
        
        # Get the graph for i and j after removing subgraphs
        start_i, start_j = random.sample(list(range(N)), 2)
        percent_i, percent_j = random.uniform(0, 0.2), random.uniform(0, 0.2)
        G_i, removed_i = remove_subgraph(molGraph.copy(), start_i, percent=percent_i)
        G_j, removed_j = remove_subgraph(molGraph.copy(), start_j, percent=percent_j)

        atom_remain_indices_i = [i for i in range(N) if i not in removed_i]
        atom_remain_indices_j = [i for i in range(N) if i not in removed_j]
        
        # Only consider bond still exist after removing subgraph
        row_i, col_i, row_j, col_j = [], [], [], []
        edge_feat_i, edge_feat_j = [], []
        G_i_edges = list(G_i.edges)
        G_j_edges = list(G_j.edges)

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
            if (start, end) in G_i_edges or (end, start) in G_i_edges:
                row_i += [start, end]
                col_i += [end, start]
                edge_feat_i.append(feature)
                edge_feat_i.append(feature)
            if (start, end) in G_j_edges or (end, start) in G_j_edges:
                row_j += [start, end]
                col_j += [end, start]
                edge_feat_j.append(feature)
                edge_feat_j.append(feature)
        
        edge_index_i = torch.tensor([row_i, col_i], dtype=torch.long)
        edge_attr_i = torch.tensor(np.array(edge_feat_i), dtype=torch.long)
        edge_index_j = torch.tensor([row_j, col_j], dtype=torch.long)
        edge_attr_j = torch.tensor(np.array(edge_feat_j), dtype=torch.long)

        ############################
        # Random Atom/Edge Masking #
        ############################

        num_mask_nodes_i = max([0, math.floor(0.25*N)-len(removed_i)])
        num_mask_edges_i = max([0, edge_attr_i.size(0)//2 - math.ceil(0.75*M)])
        num_mask_nodes_j = max([0, math.floor(0.25*N)-len(removed_j)])
        num_mask_edges_j = max([0, edge_attr_j.size(0)//2 - math.ceil(0.75*M)])
        mask_nodes_i = random.sample(atom_remain_indices_i, num_mask_nodes_i)
        mask_nodes_j = random.sample(atom_remain_indices_j, num_mask_nodes_j)
        mask_edges_i_single = random.sample(list(range(edge_attr_i.size(0)//2)), num_mask_edges_i)
        mask_edges_j_single = random.sample(list(range(edge_attr_j.size(0)//2)), num_mask_edges_j)
        mask_edges_i = [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]
        mask_edges_j = [2*i for i in mask_edges_j_single] + [2*i+1 for i in mask_edges_j_single]

        y = torch.tensor(self.labels[index], dtype=torch.long).view(1, -1)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, fingerprint=arr.reshape(1, -1))

        x_i = deepcopy(x)
        for atom_idx in range(N):
            if (atom_idx in mask_nodes_i) or (atom_idx in removed_i):
                x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_final_i = torch.zeros((2, edge_attr_i.size(0) - 2*num_mask_edges_i), dtype=torch.long)
        edge_attr_final_i = torch.zeros((edge_attr_i.size(0) - 2*num_mask_edges_i, 2), dtype=torch.long)
        count = 0
        for bond_idx in range(edge_attr_i.size(0)):
            if bond_idx not in mask_edges_i:
                edge_index_final_i[:,count] = edge_index_i[:,bond_idx]
                edge_attr_final_i[count,:] = edge_attr_i[bond_idx,:]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_final_i, edge_attr=edge_attr_final_i, fingerprint=arr.reshape(1, -1))

        x_j = deepcopy(x)
        for atom_idx in range(N):
            if (atom_idx in mask_nodes_j) or (atom_idx in removed_j):
                x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_final_j = torch.zeros((2, edge_attr_j.size(0) - 2*num_mask_edges_j), dtype=torch.long)
        edge_attr_final_j = torch.zeros((edge_attr_j.size(0) - 2*num_mask_edges_j, 2), dtype=torch.long)
        count = 0
        for bond_idx in range(edge_attr_j.size(0)):
            if bond_idx not in mask_edges_j:
                edge_index_final_j[:,count] = edge_index_j[:,bond_idx]
                edge_attr_final_j[count,:] = edge_attr_j[bond_idx,:]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_final_j, edge_attr=edge_attr_final_j, fingerprint=arr.reshape(1, -1))
        
        return data, data_i, data_j

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, test_size, data_path, target, task, splitting):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.target, self.task = target, task

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path, target=self.target, task=self.task)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.seed(0)
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=True
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=True
        )

        return train_loader, valid_loader
