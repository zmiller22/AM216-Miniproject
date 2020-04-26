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

# Read in train/test folds
test_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
train_fold=[]
for i in range(5):
    val_fold=test_fold.pop(i)
    test_fold_copy = [ee for e in test_fold for ee in e ]   
    train_fold.append(test_fold_copy)
    test_fold.insert(i,val_fold)
    
# 5 training folds are in train_fold[0]...train_fold[4], cooresponding test folds
# are in test_fold[0]...test_fold[4]

# Get bit vector representation of drugs
drugs_2 = np.array([Chem.MolFromSmiles(smile) for smile in drugs ])
drugs_ecfp2 = np.array([ Chem.GetMorganFingerprintAsBitVect(m,2) for m in drugs_2 ])

# Prepare train/test data with fold indices
rows, cols = np.where(np.isnan(affinity)==False)
 
#%% Perform 5-fold cross validation
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Flatten, Concatenate, concatenate, Dropout
from tensorflow.keras.models import Model
import gc

validate_idx = int(np.around(len(train_fold[0])*0.8))
mse_list = []

# Run 5-fold cross validation
for i in range(5):

    # Get train data for this fold
    proteins_tr = np.array([seq_to_cat(p) for p in proteins[cols[ train_fold[i][:validate_idx] ]]])[:,:,np.newaxis]#98545x1000 array of protein row vectors
    drugs_ecfp2_tr = drugs_ecfp2[ rows[ train_fold[i][:validate_idx] ] ][:,:,np.newaxis] #98545x2048 array of molecule fingerprint bit vectors
    affinity_tr = affinity[ rows[ train_fold[i][:validate_idx] ], cols[ train_fold[i][:validate_idx] ]] #98545 array of affinity vals
    
    # Get validation data for this fold
    proteins_val = np.array([seq_to_cat(p) for p in proteins[cols[ train_fold[i][validate_idx:] ]]])[:,:,np.newaxis]#98545x1000 array of protein row vectors
    drugs_ecfp2_val = drugs_ecfp2[ rows[ train_fold[i][validate_idx:] ] ][:,:,np.newaxis] #98545x2048 array of molecule fingerprint bit vectors
    affinity_val = affinity[ rows[ train_fold[i][validate_idx:] ], cols[ train_fold[i][validate_idx:] ]] #98545 array of affinity vals
    
    # get test data for this fold
    drugs_ecfp2_ts = drugs_ecfp2[ rows[test_fold[i]] ][:,:,np.newaxis]
    proteins_ts = np.array([seq_to_cat(p) for p in proteins[cols[test_fold[i]]]])[:,:,np.newaxis]
    affinity_ts = affinity[rows[test_fold[i]], cols[test_fold[i]]]

    # Define model architecture    
    drug_input_shape = (2048,1)
    protein_input_shape = (1000,1)
    
    # Conv block for the drugs
    drug_input = Input( shape=drug_input_shape, name='Drugs_Input' )
    dl_2 = Conv1D(25, 20, activation='relu')(drug_input)
    dl_3 = Conv1D(50, 20, activation='relu')(dl_2)
    dl_4  = Conv1D(100, 20, activation='relu')(dl_3)
    dl_5 = MaxPool1D(3)(dl_4)
    dl_6 = Flatten()(dl_5)
    
    # Conv block for the proteins
    protein_input = Input( shape=protein_input_shape, name="Proteins_Input")
    pl_2 = Conv1D(25, 10, activation='relu')(protein_input)
    pl_3 = Conv1D(50, 10, activation='relu')(pl_2)
    pl_4  = Conv1D(100, 10, activation='relu')(pl_3)
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
    
    # Save only the best model and stop training after an epoch with worse
    # validation loss
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=f'best_model_fold{i}.h5',
            monitor='val_loss', mode='min', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
    ]
    
    # Set training parameters
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 0.01
    MOMENTUM = 0
    OPT = tf.keras.optimizers.Adam()
    LOSS = 'mse'
    
    # Compile and train the model on fold i
    model.compile(optimizer=OPT, loss=LOSS) 
    H = model.fit([drugs_ecfp2_tr, proteins_tr], affinity_tr, 
                  validation_data=([drugs_ecfp2_val, proteins_val], affinity_val),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list,
                  verbose=1
                  )
    
    # Test the model on the test set and save the result
    model = keras.models.load_model(f'best_model_fold{i}.h5')
    score = model.evaluate([drugs_ecfp2_ts, proteins_ts], affinity_ts,
                           batch_size=64)
    mse_list.append(score)
    
    # Clear the session, grabage collect unused memory, and delete the model 
    # variable to prevent GPU memory fragmentation over loop iterations
    tf.keras.backend.clear_session()
    gc.collect()
    del model
    

#%% Train the final version of the model

# Get the training data
proteins_tr = np.array([seq_to_cat(p) for p in proteins[cols[ train_fold[0] ]]])[:,:,np.newaxis]#98545x1000 array of protein row vectors
drugs_ecfp2_tr = drugs_ecfp2[ rows[ train_fold[0] ] ][:,:,np.newaxis] #98545x2048 array of molecule fingerprint bit vectors
affinity_tr = affinity[ rows[ train_fold[0] ], cols[ train_fold[0] ]] #98545 array of affinity vals
    
# Get the validation data
drugs_ecfp2_val = drugs_ecfp2[ rows[test_fold[0]] ][:,:,np.newaxis]
proteins_val = np.array([seq_to_cat(p) for p in proteins[cols[test_fold[0]]]])[:,:,np.newaxis]
affinity_val = affinity[rows[test_fold[0]], cols[test_fold[0]]]


# Define model architecture    
drug_input_shape = (2048,1)
protein_input_shape = (1000,1)

# Conv block for the drugs
drug_input = Input( shape=drug_input_shape, name='Drugs_Input' )
dl_2 = Conv1D(25, 20, activation='relu')(drug_input)
dl_3 = Conv1D(50, 20, activation='relu')(dl_2)
dl_4  = Conv1D(100, 20, activation='relu')(dl_3)
dl_5 = MaxPool1D(3)(dl_4)
dl_6 = Flatten()(dl_5)

# Conv block for the proteins
protein_input = Input( shape=protein_input_shape, name="Proteins_Input")
pl_2 = Conv1D(25, 10, activation='relu')(protein_input)
pl_3 = Conv1D(50, 10, activation='relu')(pl_2)
pl_4  = Conv1D(100, 10, activation='relu')(pl_3)
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

# Save only the best model and stop training after 2 epochs in a row with
# worse val_loss
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath=f'final_model.h5',
        monitor='val_loss', mode='min', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
]

# Set training parameters
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.01
MOMENTUM = 0
OPT = tf.keras.optimizers.Adam()
LOSS = 'mse'

# Compile and train the model
model.compile(optimizer=OPT, loss=LOSS) 
H = model.fit([drugs_ecfp2_tr, proteins_tr], affinity_tr, 
              validation_data=([drugs_ecfp2_val, proteins_val], affinity_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list,
              verbose=1
              )
