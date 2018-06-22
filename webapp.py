import os
import sys
import csv
import json
from urlparse import urlparse

from flask import Flask, render_template, request
from flask_restful import Resource, Api
import tensorflow as tf
from extract_tfrecords_main import extract_features
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

		# Extract the features to a .tfrecord file
		print >> sys.stdout, '[SimpleVideoSearch] Extracting the features of this video'
		sys.stdout.flush()
		extract_features(tfrecord, [videoURL], ["1"], streaming = isStreamed)
		example = None
		for returnedValue in tf.python_io.tf_record_iterator(tfrecord):
			example = tf.train.Example.FromString(returnedValue)
		features = example.features
		
		# Classify the video based on the .tfrecord file and store the results in a .csv file
		print >> sys.stdout, '[SimpleVideoSearch] Classifying the video'
		infer(trained_model_dir, tfrecord, csv_file)		
		firstInference = None
		with open(csv_file, "rb") as csvfile:
			inferenceReader= csv.DictReader(csvfile)
			firstInference = inferenceReader.next()
		
		# Return a JSON containing the feature fector and inference results (for debugging purposes)
		# FIXME remove this debugging code when we're returning similar videos
		featuresJSON = MessageToJson(features)
		return '{{"feature_vector": {}, "inference_results": {}}}'.format(featuresJSON, firstInference)

		# Detect similar videos based on both the feature-vector and the classified labels
		
		# TODO In the end only return the search results (a list of videos)

api.add_resource(Videos, '/api/videos/')

if __name__ == '__main__':
    app.run(debug=True)
