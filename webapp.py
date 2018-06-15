from flask import Flask, render_template
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

@app.route('/')
def mainSearchPage():
    return render_template('index.html')

class Videos(Resource):
    def get(self):
        return {}

api.add_resource(Videos, '/api/videos/')

if __name__ == '__main__':
    app.run(debug=True)
