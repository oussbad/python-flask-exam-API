from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
min_max = MinMaxScaler()

# Load the machine learning model
model = load('mod.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract only the features that need scaling
        features_to_scale = pd.DataFrame({
            'Area': [data['Area']],
            'Area_Yards': [data['Area_Yards']]
        })

        # Apply Min-Max scaling to selected features
        scaled_features = min_max.fit_transform(features_to_scale)
        print("Scaled Features:", scaled_features)

        data['Area'], data['Area_Yards'] = scaled_features[0]

        features = [
            data['Area'], data['BHK'], data['Bathroom'], data['Furnishing'],
            data['Locality'], data['Parking'], data['Status'], data['Transaction'],
            data['Type'], data['Area_Yards']
        ]

        # Make predictions using the loaded model
        prediction = model.predict([features])
        print("Prediction before scaling:", prediction[0])

        # Scale the prediction using the same Min-Max scaler
        scaled_prediction = min_max.transform([[prediction[0], data['Area_Yards']]])
        
        # Set Content-Type header to ensure JSON response
        response = jsonify({'prediction': -scaled_prediction[0][0]})
        response.headers['Content-Type'] = 'application/json'

        return response

    except Exception as e:
        print(f"Exception: {e}")
        return jsonify({'error': f'Internal Server Error - {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
