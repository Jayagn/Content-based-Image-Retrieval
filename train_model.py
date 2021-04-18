# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 09:58:52 2021

@author: JAYAGN
"""

import cv2
import sys
import pickle
import os
import time
from tqdm import tqdm

def show_img(path):
    img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (500, 600)) 
    cv2.imshow('image',img)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_features_brute(image_path, vector_size=128,algorithm = 'SIFT'):
    
    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    if algorithm == 'SIFT':
        alg = cv2.SIFT_create()
    elif algorithm == 'KAZE':
        alg = cv2.KAZE_create()
    elif algorithm == 'ORB':
        alg = cv2.ORB_create()
    #Detecting Keypoints    
    keypoints = alg.detect(image)
    #Number of keypoints varies depending upon images, hence getting first 128 Keypoints of each reference image.
    # Sorting them based on keypoint response value (bigger is considered to be better)
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:vector_size]
    #Extracting descriptors
    keypoints, descriptors = alg.compute(image, keypoints)
    
    return keypoints, descriptors

def batch_extractor_brute(images_path, algorithm = 'SIFT'):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    names = []
    keypoints= []
    descriptors = []
    for f in tqdm(files,position=0, leave=True):
        names.append(f)
        keypoint, descriptor = extract_features_brute(f,128,algorithm)
        keypoints.append(keypoint)
        descriptors.append(descriptor)
    
    return names, keypoints, descriptors

def find_good_match(ref_descriptors,ref_names,des_query,query_image_path):
    
    #Using FLANN based matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    #Creating list of good matches
    good_matches_list = []
    
    #Performing KNN match on each os reference image descriptors and given query image descriptor
    for i in range(len(ref_descriptors)):
        matches = flann.knnMatch(ref_descriptors[i], des_query, k=2)

        #Storing only good matches as per Lowe's ratio test.
        good_matches = 0
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good_matches+=1
        good_matches_list.append(good_matches)
        
    print("Query Image:")
    show_img(query_image_path)
    print("Matched Reference Image")
    #Index of reference image with maximum number of good matches is retrieved and displayed.
    show_img(ref_names[good_matches_list.index(max(good_matches_list))])


if __name__ == "__main__":  
    #Provide algorithm to extract features
    algorithm ='SIFT' #SIFT or KAZE
    
    #Storing names(index), keypoints and descriptor of reference images as pickle file
    pickle_names = 'pickle_names_'+algorithm+'.pck'
    pickle_keypoints = 'pickle_keypoints_'+algorithm+'.pck'
    pickle_descriptor = 'pickle_descriptor_'+algorithm+'.pck'
    
    #Path to folder containing reference images
    reference_images_path = 'referenceImages'
    
    #Calculate time taken by algorithm to extract features
    start = time.time()
    ref_names, ref_keypoints, ref_descriptors = batch_extractor_brute(reference_images_path, algorithm)
    end= time.time()
    print(f"Runtime of the program is {end - start}")
    
    with open(pickle_names, 'wb') as fb:
        pickle.dump(ref_names, fb)
    with open(pickle_descriptor, 'wb') as ab:
        pickle.dump(ref_descriptors, ab)
    
