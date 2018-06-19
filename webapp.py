import os
import sys
from urlparse import urlparse

from flask import Flask, render_template, request
from flask_restful import Resource, Api
from extract_tfrecords_main import extract_features
import tensorflow as tf

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
		# return interim results (extracted features)
		return '{}'.format(retval)
		# TODO In the end only return the search results (a list of videos)
		#return {'url': videoURL}

api.add_resource(Videos, '/api/videos/')

if __name__ == '__main__':
    app.run(debug=True)
