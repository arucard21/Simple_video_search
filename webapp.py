import os
import sys
import csv
import json
from urlparse import urlparse
from datetime import datetime

from flask import Flask, render_template, request
from flask_restful import Resource, Api
import tensorflow as tf
from youtube8m.feature_extractor.extract_tfrecords_main import extract_features
from similarity import load_forest, similar_videos, similar_videos_from_forest
from youtube8m.inference import infer
from google.protobuf.json_format import MessageToJson

app = Flask(__name__)
api = Api(app)

@app.route('/')
def mainSearchPage():
    return render_template('index.html')

class Videos(Resource):
    def get(self):
		# define variables for file and directory names
		tfrecord = "providedVideo.tfrecord"
		csv_file = "infer_results.csv"
		trained_model_dir = "logistic-model"

		# parse URL and check if the hostname is that of a streaming site
		videoURL = request.args.get('videoURL')
		url = urlparse(videoURL)
		isStreamed = False
		if url.netloc == 'www.youtube.com':
			isStreamed = True
		
		
		useForestParam = request.args.get('useForest')
		useForest = False
		if useForestParam.lower() == 'true':
			useForest = True

		# Extract the features to a .tfrecord file
		print >> sys.stdout, '[SimpleVideoSearch][{}] Extracting the features of this video'.format(datetime.now())
		sys.stdout.flush()
		extract_features(tfrecord, [videoURL], ["1"], streaming = isStreamed)
		example = None
		for returnedValue in tf.python_io.tf_record_iterator(tfrecord):
			example = tf.train.Example.FromString(returnedValue)
		features = example.features

		# Classify the video based on the .tfrecord file and store the results in a .csv file
		print >> sys.stdout, '[SimpleVideoSearch][{}] Classifying the video'.format(datetime.now())
		infer(trained_model_dir, tfrecord, csv_file)		
		firstInference = None
		with open(csv_file, "rb") as csvfile:
			inferenceReader= csv.DictReader(csvfile)
			firstInference = inferenceReader.next()

		if useForest:
			print >> sys.stdout, '[SimpleVideoSearch][{}] Searching using the pickled forest'.format(datetime.now())
			return similar_videos_from_forest(firstInference)
		else:
			print >> sys.stdout, '[SimpleVideoSearch][{}] Searching directly in the dataset'.format(datetime.now())
			top10_feature_based, top10_label_based = similar_videos(features, firstInference)
			return [top10_feature_based, top10_label_based]

api.add_resource(Videos, '/api/videos/')

if __name__ == '__main__':
	load_forest()
	app.run(debug=True, use_reloader=False)
