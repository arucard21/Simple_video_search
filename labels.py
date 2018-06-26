import glob
import pickle
from datetime import datetime
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import jaccard
from datasketch import MinHashLSHForest, MinHash
LABELS_FILE = 'labels.pkl'
MAX_AMOUNT_LABELS = 4716
labels = None

def get_labels(video_id):
	return labels[video_id]

def load_labels():
	global labels
	print "[SimpleVideoSearch][{}] Loading pickled labels dict".format(datetime.now())
	with open(LABELS_FILE, 'rb') as labels_file:
		labels = pickle.load(labels_file)
	print "[SimpleVideoSearch][{}] Done loading pickled labels dict".format(datetime.now())

def create_label_datastore():
	train_records = glob.glob("dataset/train*.tfrecord")
	validate_records = glob.glob("dataset/validate*.tfrecord")
	all_records = train_records+validate_records
	dataset = tf.data.TFRecordDataset(all_records)
	iterator = dataset.make_one_shot_iterator()
	labels = {}
	count = 0
	next_element = iterator.get_next()
	with tf.Session() as sess:
		try:
			while True:
				if count % 100000 == 0:
					print "[SimpleVideoSearch][{}] Processed {} records from the dataset so far".format(datetime.now(), count)
				exampleBinaryString= sess.run(next_element)
				example = tf.train.Example.FromString(exampleBinaryString)
				count += 1
				example_id = example.features.feature["id"].bytes_list.value[0]
				example_labels = list(example.features.feature["labels"].int64_list.value)
				labels[example_id] = example_labels
		except tf.errors.OutOfRangeError:
			print "[SimpleVideoSearch][{}] Done iterating through dataset".format(datetime.now())
		finally:
			print "[SimpleVideoSearch][{}] Processed {} records from the dataset".format(datetime.now(), count)
		with open(LABELS_FILE, 'wb') as labels_file:
			pickle.dump(labels, labels_file, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	create_label_datastore()
