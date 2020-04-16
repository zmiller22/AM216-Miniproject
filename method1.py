#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import deepchem as dc
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import MolFromSmiles
from deepchem.feat import RDKitDescriptors   
import networkx as nx

import tensorflow as tf


#%% TF's code from project notebook (cut out unimportant parts)

# for converting protein sequence to categorical format
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:i for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000   # Note that all protein data will have the same length 1000 

def seq_to_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  


fpath = 'data/kiba/'

# Read in drugs and proteins
drugs_ = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
drugs = np.array([ Chem.MolToSmiles(Chem.MolFromSmiles(d),isomericSmiles=True) for d in drugs_.values() ])
proteins_ = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
proteins = np.array(list(proteins_.values()))

# Read in affinity data
affinity = np.array(pickle.load(open(fpath + "Y","rb"), encoding='latin1'))

# Read in train/test fold  
train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
train_fold = [ee for e in train_fold for ee in e ]    
'''
Here all validation folds are aggregated into training set. 
If you want to train models with different architectures and/or 
optimize for model hyperparameters, we encourage you to use 5-fold 
cross validation as provided here.
'''
test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))



# Prepare train/test data with fold indices
rows, cols = np.where(np.isnan(affinity)==False)
 
drugs_tr = drugs[ rows[train_fold] ] #98545 array of SMILES strings
proteins_tr = np.array([ seq_to_cat(p) for p in proteins[cols[train_fold]] ])#98545x1000 array of protein row vectors
affinity_tr = affinity[rows[train_fold], cols[train_fold]] #98545 array of affinity vals

drugs_ts = drugs[rows[test_fold]]
proteins_ts = np.array([seq_to_cat(p) for p in proteins[cols[test_fold]]])
affinity_ts = affinity[rows[test_fold], cols[test_fold]]  

#%% Get data ready for ML models
drugs_2 = np.array([Chem.MolFromSmiles(smile) for smile in drugs ])
drugs_ecfp2 = np.array([ Chem.GetMorganFingerprintAsBitVect(m,2) for m in drugs_2 ])
drugs_ecfp2_tr = drugs_ecfp2[ rows[train_fold] ] #98545x2048 array of molecule fingerprint bit vectors
drugs_ecfp2_ts = drugs_ecfp2[ rows[test_fold] ]

# Concatenate the protein and drugs vectors into one array
train_data = np.hstack( (proteins_tr, drugs_ecfp2_tr) )
train_vals = affinity_tr
test_data = np.hstack( (proteins_ts, drugs_ecfp2_ts) )
test_vals = affinity_ts

#%%