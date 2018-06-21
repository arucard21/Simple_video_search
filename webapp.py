import os
import sys
import csv
from urlparse import urlparse

from flask import Flask, render_template, request
from flask_restful import Resource, Api
import tensorflow as tf
from extract_tfrecords_main import extract_features
from youtube8m.inference import infer

app = Flask(__name__)
api = Api(app)

@app.route('/')
def mainSearchPage():
    return render_template('index.html')

class Videos(Resource):
    def get(self):
		videoURL = request.args.get('videoURL')
		url = urlparse(videoURL)
		isStreamed = False
		if url.netloc == 'www.youtube.com':
			isStreamed = True
		tfrecord = "providedVideo.tfrecord"
		print >> sys.stdout, '[SimpleVideoSearch] Extracting the features of this video'
		sys.stdout.flush()
		extract_features(tfrecord, [videoURL], ["1"], streaming = isStreamed)
		retval = ""
		for returnedValue in tf.python_io.tf_record_iterator('providedVideo.tfrecord'):
			retval = tf.train.Example.FromString(returnedValue)
		sys.stdout.flush()
		
		print >> sys.stdout, '[SimpleVideoSearch] Classifying the video'
		infer("logistic-model", "providedVideo.tfrecord", "infer_results.csv")
		
		firstInference = None
		with open("infer_results.csv", "rb") as csvfile:
			inferenceReader= csv.DictReader(csvfile)
			firstInference = inferenceReader.next()
		return firstInference

		# return interim results (extracted features)
		#return '{}'.format(retval)
		# TODO In the end only return the search results (a list of videos)
		#return {'url': videoURL}

api.add_resource(Videos, '/api/videos/')

if __name__ == '__main__':
    app.run(debug=True)
