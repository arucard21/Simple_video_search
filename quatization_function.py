# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 09:02:38 2018

@author: rmaan
"""

import numpy as np

def quantization(dataset, threshold):
    # check total number of videos and number of feature
    [total_videos, num_features] = dataset.shape
    
    # iterate over all videos
    for i in range(1,total_videos):
        for j in range(1,num_features):
            # set value based on threshold
            if(dataset[i,j] >= threshold):
                dataset[i,j] = 1
            else:
                dataset[i,j] = 0
    
    return dataset