from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('iris_flower_model.pkl')

species_dict = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.get_json()
 
        if not all(feature in data for feature in ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']):
            return jsonify({'error': 'Missing features in the input data'}), 400

        features = [data['sepalLength'], data['sepalWidth'], data['petalLength'], data['petalWidth']]

        prediction = model.predict([features])
        species = species_dict.get(prediction[0], 'Unknown')

        return jsonify({'species': species})

    except Exception as e:
 
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

