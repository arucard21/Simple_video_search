import glob
import numpy as np
import tensorflow as tf

def nearest_neighbor_single_feature(provided, dataset):
	assert len(provided) == len(dataset)
	distance = 0.0
	for index in range(len(provided)):
		distance = distance + abs(provided[index] - dataset[index])
	return distance

def nearest_neighbor_features(provided, dataset):
	#Compare for both the mean_rgb and mean_audio features (the only 2 available)
	provided_mean_rgb = provided.feature["mean_rgb"].float_list.value
	dataset_mean_rgb = dataset.feature["mean_rgb"].float_list.value

	provided_mean_audio = provided.feature["mean_audio"].float_list.value
	dataset_mean_audio = dataset.feature["mean_audio"].float_list.value	

	distance_mean_rgb = nearest_neighbor_single_feature(provided_mean_rgb, dataset_mean_rgb)
	distance_mean_audio = nearest_neighbor_single_feature(provided_mean_audio, dataset_mean_audio)

	return distance_mean_rgb + distance_mean_audio

def similar_videos(provided_features, inferred_label_probabilities):
	train_records = glob.glob("dataset/train00.tfrecord") # FIXME change back to train*.tfrecord
	validate_records = glob.glob("dataset/validate00.tfrecord") # FIXME change back to validate*.tfrecord
	all_records = train_records+validate_records
	dataset = tf.data.TFRecordDataset(all_records)
	iterator = dataset.make_one_shot_iterator()

	count = 0
	# a list of tuples containing the id and nearest-neighbor distance for each element, using the features
	nn_distance = list()
	# a list of tuples containing the id and Jaccard distance for each element, using the inferred label probabilities (not used yet)
	jac_distance = list()
	next_element = iterator.get_next()
	with tf.Session() as sess:
		try:
			while True:
				exampleBinaryString= sess.run(next_element)
				example = tf.train.Example.FromString(exampleBinaryString)
				count += 1
				# Compare the provided features with this element of the dataset (nearest neighbor)
				nn_distance.append((example.features.feature["id"].bytes_list.value[0], nearest_neighbor_features(provided_features, example.features)))
				# TODO compare the provided inference results with this element of the dataset (Jaccard distance)
				# the distances should be put in the jac_distance variable
		except tf.errors.OutOfRangeError:
			print "Done iterating through dataset"
		finally:
			print "Processed {} records from the dataset".format(count)
	# Sort the lists based on distance
	nn_distance.sort(key = lambda tuple: tuple[1])
	jac_distance.sort(key = lambda tuple: tuple[1])
	# Get the top 10 results
	top10_feature_based = nn_distance[:10]
	top10_label_based = jac_distance[:10]
	return (top10_feature_based, top10_label_based)
