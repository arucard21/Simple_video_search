import numpy as np

def nearest_neighbors(video_features, dataset):
    # check total number of videos and number of feature
    [total_videos, num_features] = dataset.shape
    # make a distance array to store distance of given video to all others 
    dist = np.zeros(total_videos)
    # iterate over all videos
    for i in range(1,total_videos):
        for j in range(1,num_features):
            dist[1,i] = dist[1,i] + (video_features[1,j]-dataset[i,j])^2
    # return indices of videos in descending order of distance
    indices = np.flipud(np.argsort(dist))
    return indices