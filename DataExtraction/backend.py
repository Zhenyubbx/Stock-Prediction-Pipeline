from flask import Flask, request
from sentiment import test_predict_with_linear_model
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # # Get the parameters from the request
    # param1 = request.json['param1']
    # param2 = request.json['param2']
    
    # Call the function with the parameters
    result = test_predict_with_linear_model()
    print(result)
    # result = "hello world"
    
    # Return the result as JSON
    return {'result': result}

if __name__ == '__main__':
    app.run()