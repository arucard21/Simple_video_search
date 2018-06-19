import os

from flask import Flask, render_template, request
from flask_restful import Resource, Api
from extract_tfrecords_main import extract_features

app = Flask(__name__)
api = Api(app)

@app.route('/')
def mainSearchPage():
    return render_template('index.html')

class Videos(Resource):
    def get(self):
		videoURL = request.args.get('videoURL')
		tfrecord = "providedVideo.tfrecord"
		extract_features(tfrecord, [videoURL], ["1"])
		# show interim results
		if os.path.getsize(tfrecord) > 0:
			return 'The video at {} successfully had its features extracted'.format(videoURL) 
		else:
			return 'The video at {} failed to have its features extracted'.format(videoURL) 
		
		# TODO In the end only return the search results (a list of videos)
		#return {'url': videoURL}

api.add_resource(Videos, '/api/videos/')

if __name__ == '__main__':
    app.run(debug=True)
