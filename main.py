# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 10:05:53 2021

@author: JAYAGN
"""
import sys
import os
import pickle
from train_model import extract_features_brute, find_good_match

def main():
    
    algorithm = 'SIFT' #Remember to give same algorithm used extracting features of reference images.
    
    names_path = 'pickle_names_'+algorithm+'.pck'
    descriptor_path = 'pickle_descriptor_'+algorithm+'.pck'
    
    #Load the saved model
    with open(names_path, 'rb') as db: 
        reloaded_names = pickle.load(db)
    with open(descriptor_path, 'rb') as eb: 
        reloaded_descriptor = pickle.load(eb)
    
    #Give command line input as path to query image
    query_path= sys.argv[1]    
    query_images_path = query_path
    
    #Extract features of query image.
    kp_query, des_query = extract_features_brute(query_images_path,128,algorithm)
    
    find_good_match(reloaded_descriptor,reloaded_names,des_query,query_images_path)

main()