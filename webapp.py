from flask import Flask, render_template, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

@app.route('/')
def mainSearchPage():
    return render_template('index.html')

class Videos(Resource):
    def get(self):
		videoURL = request.args.get('videoURL')
		# TODO download the video from the URL, process it and return the list of similar videos
		return {'url': videoURL}

api.add_resource(Videos, '/api/videos/')

if __name__ == '__main__':
    app.run(debug=True)
