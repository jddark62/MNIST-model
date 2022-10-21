import numpy as np
import pickle
import matplotlib.pyplot as plt
from flask import *


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def accuracy(predictions, Y):
    return np.mean(predictions == Y)


def predict(X, W1, W2, b1, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(A2, 0)


(W1, W2, b1, b2) = pickle.load(open("model.p", "rb"))

app = Flask(__name__)


@app.route('/static')
def send_report(path):
    return send_from_directory('static', path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def route_predict():
    p = np.array([request.json['input']])
    pred = predict(p.T, W1, W2, b1, b2)
    return jsonify({'prediction': str(pred[0])})

if __name__ == "__main__":
    app.run(debug=True)
