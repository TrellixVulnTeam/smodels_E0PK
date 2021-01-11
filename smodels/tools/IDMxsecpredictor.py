#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Humberto Reyes-Gonzalez
"""

import numpy as np
import pyslha
import tensorflow as tf




def fetch_input(slhafile):

	slha_read= pyslha.read(slhafile)

	input_list=[]
         
         MH0 = slha_read.blocks['MASS'][35]
         MA0 = slha_read.blocks['MASS'][36]
         MHC = slha_read.blocks['MASS'][37]
         lam2 = slha_read.blocks['FRBLOCK'][6]
         lamL = slha_read.blocks['FRBLOCK'][5]
         
         
         input_list.get('MH0').append(MH0)
         input_list.get('MA0').append(MA0)
         input_list.get('MHC').append(MHC)
         input_list.get('lam2').append(lam2)
         input_list.get('lamL').append(lamL)

	

	return input_list

	

def preprocess(X):
    
    
    ##inputs, z-score

    ###NOTE The mean should be fixed. Fetch this!
    
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0)
    
    return X

    
def postprocess(preds, samples):


    preds = preds + np.log(10**(-7))
    preds = np.exp(preds)
    print(np.shape(preds))
    return preds

def load_models(process):

	model=tf.keras.models.load_model(process+'.hdf5')
	
	return model
	


def predictor(model,X):
    
    
    preds = []
    for i in range(0, samples):
        y_pred = model.predict(X)
        y_pred = np.array(y_pred)
        y_pred = np.reshape(y_pred, (-1,))
        preds.append(y_pred)
        
    
    preds = np.array(preds)
    y_pred_post = postprocess(preds, samples)
        

    mean_y_pred = np.mean(y_pred_post) 
    std_y_pred =np.std(y_pred_post)   


    return mean_y_pred, std_y_pred
 
         
         
 def predict():

	input_list=fetch_input(slhafile)
	X=preprocess(input_list)
	
	predictions={'3535':[],'3536':[],'3537':[],'3636':[],'3637':[],'3735':[],'3736':[],'3737':[]}
	for process in predictions.keys():
		
		model=load_models(process)
		mean_pread,std_pred=predictor(model,X)

		predcitions.get(process).append(mean_pread)
		predcitions.get(process).append(std_pread)

	return predictions
		

	   
    

        
    

    

    
