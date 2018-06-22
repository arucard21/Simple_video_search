import glob
import numpy as np
import tensorflow as tf

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

def similar_videos(feature_vector, inference_results):
	train_records = glob.glob("dataset/train*.tfrecord")
	validate_records = glob.glob("dataset/validate*.tfrecord")
	all_records = train_records+validate_records
	#filenames = ["dataset/train00.tfrecord", "dataset/validate00.tfrecord"]
	dataset = tf.data.TFRecordDataset(all_records)
	iterator = dataset.make_one_shot_iterator()
	count = 0
	next_element = iterator.get_next()
	with tf.Session() as sess:
		try:
			while True:
				exampleBinaryString= sess.run(next_element)
				example = tf.train.Example.FromString(exampleBinaryString)
				count += 1
				
				# TODO compare the feature vector with this element of the dataset
				
				# TODO compare the inference results with this element of the dataset
				
				#print example.features.feature["id"]
		except tf.errors.OutOfRangeError:
			print "Done iterating through dataset"
		finally:
			print "Processed {} records from the dataset".format(count)

if __name__ == '__main__':
    similar_videos(None, None)
