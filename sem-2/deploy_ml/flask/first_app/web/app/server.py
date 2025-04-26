from flask import Flask, request, jsonify
import pickle
import numpy as np

# загружаем модель из файла
with open('./models/model.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)

# создаём приложение
app = Flask(__name__)

@app.route('/')
def index():
    msg = "Test message. The server is running"
    return msg

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json
    features = np.array(features).reshape(1, 4)
    pred = model.predict(features)[0]
    res = {'prediction': pred}
    return jsonify(res)

if __name__ == '__main__':
    app.run('0.0.0.0', 5000)