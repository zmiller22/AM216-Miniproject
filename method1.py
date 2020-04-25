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
train_fold = json.load(open(fpath + "train_fold_setting1.txt"))
cross_val_train=[]
for i in range(5):
    val_fold=train_fold.pop(i)
    train_fold_copy = [ee for e in train_fold for ee in e ]   
    cross_val_train.append(train_fold_copy)
    train_fold.insert(i,val_fold)
    
'''
cross validation folds are stored in train_fold[0]...train_fold[4]
the corresponding training folds are stored in cross_val_train[0]...cross_val_train[4]
'''
test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))

# Get bit vector representation of drugs
drugs_2 = np.array([Chem.MolFromSmiles(smile) for smile in drugs ])
drugs_ecfp2 = np.array([ Chem.GetMorganFingerprintAsBitVect(m,2) for m in drugs_2 ])


# Prepare train/test data with fold indices
rows, cols = np.where(np.isnan(affinity)==False)
 
#%%
# Create test and train data
drugs_tr = drugs[ rows[train_fold] ] #98545 array of SMILES strings
proteins_tr = np.array([ seq_to_cat(p) for p in proteins[cols[train_fold]] ])[:,:,np.newaxis]#98545x1000 array of protein row vectors
drugs_ecfp2_tr = drugs_ecfp2[ rows[train_fold] ][:,:,np.newaxis] #98545x2048 array of molecule fingerprint bit vectors
affinity_tr = affinity[rows[train_fold], cols[train_fold]] #98545 array of affinity vals


drugs_ts = drugs[rows[test_fold]]
drugs_ecfp2_ts = drugs_ecfp2[ rows[test_fold] ][:,:,np.newaxis]
proteins_ts = np.array([seq_to_cat(p) for p in proteins[cols[test_fold]]])[:,:,np.newaxis]
affinity_ts = affinity[rows[test_fold], cols[test_fold]]

## In case we decide to use concatenated vectors
# # Concatenate the protein and drugs vectors into one array
# train_data = np.hstack( (proteins_tr, drugs_ecfp2_tr) )
# train_vals = affinity_tr
# test_data = np.hstack( (proteins_ts, drugs_ecfp2_ts) )
# test_vals = affinity_ts

#%% Using tf.keras api to create a custom model shape (not sequential)
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Flatten, Concatenate, concatenate, Dropout
from tensorflow.keras.models import Model

drug_input_shape = (drugs_ecfp2_tr.shape[1],1)
protein_input_shape = (proteins_tr.shape[1],1)

# Conv block for the drugs
drug_input = Input( shape=drug_input_shape, name='Drugs_Input' )
dl_2 = Conv1D(50, 20, activation='relu')(drug_input)
dl_3 = Conv1D(100, 20, activation='relu')(dl_2)
dl_4  = Conv1D(150, 20, activation='relu')(dl_3)
dl_5 = MaxPool1D(3)(dl_4)
dl_6 = Flatten()(dl_5)

# Conv block for the proteins
protein_input = Input( shape=protein_input_shape, name="Proteins_Input")
pl_2 = Conv1D(50, 10, activation='relu')(protein_input)
pl_3 = Conv1D(100, 10, activation='relu')(pl_2)
pl_4  = Conv1D(150, 10, activation='relu')(pl_3)
pl_5 = MaxPool1D(3)(pl_4)
pl_6 = Flatten()(pl_5)

# Combined output of both the Conv blocks
comb_input = Concatenate(1)([dl_6, pl_6])

# Dense layers to output
cl_1 = Dense(512)(comb_input)
cl_2 = Dropout(0.1)(cl_1)
cl_3 = Dense(512)(cl_2)
cl_4 = Dropout(0.1)(cl_3)
cl_5 = Dense(256)(cl_4)
output = Dense(1)(cl_5)

# Create the model
model = Model(inputs=[drug_input, protein_input], outputs=output)

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
]

# Training parameters
BATCH_SIZE = 128
EPOCHS = 50

# Optimizer Parameters
LR = 0.01
MOMENTUM = 0
OPT = tf.keras.optimizers.Adam()
#OPT = tf.keras.optimizers.SGD(learning_rate=LR, momentum=MOMENTUM)

# Loss parameters
#LOSS = tf.losses.mean_squared_error()

model.compile(optimizer=OPT, loss='mse') 
H = model.fit([drugs_ecfp2_tr, proteins_tr], affinity_tr, 
              validation_data=([drugs_ecfp2_ts, proteins_ts], affinity_ts),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list,
              verbose=1
              )


