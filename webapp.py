import os
import sys
import csv
import json
from urllib.parse import urlparse
from datetime import datetime

from flask import Flask, render_template, request
from flask_restful import Resource, Api
import tensorflow as tf
from youtube8m.feature_extractor.extract_tfrecords_main import extract_features
from similarity import load_forest, similar_videos, similar_videos_from_forest, get_inferred_labels
from labels import load_labels, get_labels
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
		print('[SimpleVideoSearch][{}] Extracting the features of this video'.format(datetime.now()), file=sys.stdout)
		sys.stdout.flush()
		extract_features(tfrecord, [videoURL], ["1"], streaming = isStreamed)
		example = None
		for returnedValue in tf.python_io.tf_record_iterator(tfrecord):
			example = tf.train.Example.FromString(returnedValue)
		features = example.features

		# Classify the video based on the .tfrecord file and store the results in a .csv file
		print('[SimpleVideoSearch][{}] Classifying the video'.format(datetime.now()), file=sys.stdout)
		infer(trained_model_dir, tfrecord, csv_file)		
		firstInference = None
		with open(csv_file, "rb") as csvfile:
			inferenceReader= csv.DictReader(csvfile)
			firstInference = next(inferenceReader)

		if useForest:
			print('[SimpleVideoSearch][{}] Searching using the pickled forest'.format(datetime.now()), file=sys.stdout)
			return similar_videos_from_forest(firstInference)
		else:
			print('[SimpleVideoSearch][{}] Searching directly in the dataset'.format(datetime.now()), file=sys.stdout)
			top10_feature_based, top10_label_based = similar_videos(features, firstInference)
			return [top10_feature_based, top10_label_based]

class Labels(Resource):
	def get(self, video_id):
		return get_labels(video_id)

class InferredLabels(Resource):
	def get(self, video_id):
		return get_inferred_labels(video_id)
	
api.add_resource(Videos, '/api/videos/')
api.add_resource(Labels, '/api/labels/<string:video_id>')
api.add_resource(InferredLabels, '/api/inferred/<string:video_id>')

if __name__ == '__main__':
	load_labels()
	load_forest()
	app.run(debug=True, use_reloader=False)
