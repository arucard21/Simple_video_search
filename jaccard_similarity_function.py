# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 08:48:46 2018

@author: rmaan
"""

import numpy as np
from quantization_function import quantization

def jaccard_similarity(video_features, dataset):
    # check total number of videos and number of feature
    [total_videos, num_features] = dataset.shape
    
    # qunatize dataset and video_features based on threshold
    threshold = 0.6
    dataset = quantization(dataset, threshold)
    video_features = quantization(video_features, threshold)
    
    # make a similarity array to store similarity of given video to all others 
    sim = np.zeros(total_videos)
    # make an intersection array 
    intersection = np.zeros(total_videos)
    # make a union array
    union = np.zeros(total_videos)
    # iterate over all videos
    for i in range(1,total_videos):
        for j in range(1,num_features):
            # if video and dataset_video is 1, then, add 1 to intersection
            if(dataset[i,j] == 1 and video_features[1,j]==1):
                intersection[1,i] = intersection[1,i] + 1;
                union[1,i] = union[1,i] + 1;
            # if one of video or datatset_video is 1, then, add 1 to union
            elif(dataset[i,j]==1 or video_features[1,j]==1):
                union[1,i] = union[1,i] + 1;
    
    # calculate jaccard similarity for all videos
    for i in range(1,total_videos):
        sim[1,i] = intersection[1,i]/union[1,i]
                
    # return indices of videos in descending order of similarity
    indices = np.flipud(np.argsort(sim))
    return indices