from flask import Flask, render_template
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

@app.route('/')
def mainSearchPage():
    return render_template('index.html')

class Features(Resource):
    def get(self):
        return {}

class Keywords(Resource):
    def get(self):
        return {}

class Videos(Resource):
    def get(self):
        return {}

api.add_resource(Features, '/api/features/')
api.add_resource(Keywords, '/api/keywords/')
api.add_resource(Videos, '/api/videos/')

if __name__ == '__main__':
    app.run(debug=True)
