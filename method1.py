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
import tensorflow.keras as keras


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

# Get bit vector representation of drugs
drugs_2 = np.array([Chem.MolFromSmiles(smile) for smile in drugs ])
drugs_ecfp2 = np.array([ Chem.GetMorganFingerprintAsBitVect(m,2) for m in drugs_2 ])


# Prepare train/test data with fold indices
rows, cols = np.where(np.isnan(affinity)==False)
 

# Create test and train data
drugs_tr = drugs[ rows[train_fold] ] #98545 array of SMILES strings
proteins_tr = np.array([ seq_to_cat(p) for p in proteins[cols[train_fold]] ])#98545x1000 array of protein row vectors
affinity_tr = affinity[rows[train_fold], cols[train_fold]] #98545 array of affinity vals
drugs_ecfp2_tr = drugs_ecfp2[ rows[train_fold] ] #98545x2048 array of molecule fingerprint bit vectors

drugs_ts = drugs[rows[test_fold]]
drugs_ecfp2_ts = drugs_ecfp2[ rows[test_fold] ]
proteins_ts = np.array([seq_to_cat(p) for p in proteins[cols[test_fold]]])
affinity_ts = affinity[rows[test_fold], cols[test_fold]]  

## In case we decide to use concatenated vectors
# # Concatenate the protein and drugs vectors into one array
# train_data = np.hstack( (proteins_tr, drugs_ecfp2_tr) )
# train_vals = affinity_tr
# test_data = np.hstack( (proteins_ts, drugs_ecfp2_ts) )
# test_vals = affinity_ts

#%% Using tf.keras api to create a custom model shape (not sequential)
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Flatten, Concatenate, concatenate
from tensorflow.keras.models import Model

drug_input_shape = (drugs_ecfp2_tr.shape[1],1)
protein_input_shape = (drugs_ecfp2_tr.shape[1],1)

dl_1 = Input( shape=drug_input_shape, name='Drugs_Input' )
dl_2 = Conv1D(100, 20, activation='relu')(dl_1)
dl_3 = Conv1D(150, 20, activation='relu')(dl_2)
dl_4  = Conv1D(200, 20, activation='relu')(dl_3)
dl_5 = MaxPool1D(3)(dl_4)
dl_6 = Flatten()(dl_5)

pl_1 = Input( shape=protein_input_shape, name="Proteins_Input")
pl_2 = Conv1D(100, 10, activation='relu')(pl_1)
pl_3 = Conv1D(150, 10, activation='relu')(pl_2)
pl_4  = Conv1D(200, 10, activation='relu')(pl_3)
pl_5 = MaxPool1D(3)(pl_4)
pl_6 = Flatten()(pl_5)

#comb_input = tf.stack([dl_5, pl_5])
comb_input = Concatenate(1)([dl_6, pl_6])

test_model = Model(inputs=[dl_1, pl_1], outputs=comb_input)
