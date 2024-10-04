from flask import request, jsonify, make_response
import random
from dotenv import load_dotenv

import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from datetime import datetime
import random

import pickle

load_dotenv()

def predict_flood():
    try:
        data = request.get_json()
        name = data.get('name')
        input_data = data.get('data')  # Renamed to avoid overwriting the `data` variable
        
        if not name or not input_data:
            raise ValueError("All fields are required")

        response = make_response(jsonify({
            "success": True,
            "message": "Model predicted successfully",
            "response": {
                "name": name,
                "data": input_data,
                "random_number": random.randint(1, 100),  # Fixed randint call
            }
        }), 201)
        
        return response
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred"}), 500




def prediction_drought(date):
    try:
        
        with open('server/controllers/Drought_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        predictions = model.forecast(steps=1)
        # predictions = model.predict(date)
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise  


# def predict_air_quality():
#     try:
#         date = request.json.get('date')
#         if not date:
#             return jsonify({"error": "Date is required"}), 400
        
#         predictions = prediction(date)  
        
#         predictions_dict = {str(k): v for k, v in predictions.items()}
        
        
#         return jsonify({'response': predictions_dict})
#     except Exception as e:
#         print(f"Error in route: {str(e)}")
#         return jsonify({"error": str(e)}), 400

def predict_drought():
    try:
        # date = request.json.get('date')
        # if not date:
        #     return jsonify({"error": "Date is required"}), 400
        date = 12
        predictions = prediction_drought(date)  
        
        predictions_dict = {str(k): v for k, v in predictions.items()}
        
        
        return jsonify({'response': predictions_dict})
    except Exception as e:
        print(f"Error in route: {str(e)}")
        return jsonify({"error": str(e)}), 400




def update_csv(file_path, data):
    
    df = pd.read_csv(file_path)
    
    new_date = datetime.today().strftime('%Y-%m-%d')
    latitude = 42.15443863927893 
    longitude = -119.16159733285662 
    soil_moisture = data 
    
    # Create the new row as a dictionary
    new_row = {'Date': new_date, 'Latitude': latitude, 'Longitude': longitude, 'Soil_Moisture': soil_moisture}
    
    new_row_df = pd.DataFrame([new_row])
    
    df = pd.concat([df[1:], new_row_df], ignore_index=True) 
    
    df.to_csv(file_path, index=False)
    # print(f"Updated file saved with new value: {new_row}")


def preprocess_input(new_data, scaler, window_size=195, features=['Soil_Moisture', 'sin_day', 'cos_day', 'month'], date_format='%Y-%m-%d'):
    """
    Preprocesses the input data for the LSTM model.
    Parameters:
    - new_data (pd.DataFrame): The complete dataset including the latest records.
    Must contain 'Date', 'Soil_Moisture', 'Latitude', 'Longitude' columns.
    - scaler (MinMaxScaler): The scaler fitted on the training data.
    - window_size (int): Number of past records to use for prediction.
    - features (list): List of feature column names to be used.
    - date_format (str): The format of the 'Date' column in new_data (e.g., '%Y-%m-%d').
    Returns:
    - np.ndarray: The preprocessed data reshaped to (1, window_size, num_features).
    """
    # Parse 'Date' column to datetime and set as index
    new_data['Date'] = pd.to_datetime(new_data['Date'], format=date_format)
    new_data.set_index('Date', inplace=True)
    
    # Sort the data by date in ascending order
    new_data.sort_index(inplace=True)
    
    # Feature Engineering
    new_data['day_of_year'] = new_data.index.dayofyear
    new_data['sin_day'] = np.sin(2 * np.pi * new_data['day_of_year'] / 365.25)
    new_data['cos_day'] = np.cos(2 * np.pi * new_data['day_of_year'] / 365.25)
    new_data['month'] = new_data.index.month
    
    # Select relevant features
    df_features = new_data[features]
    
    # Check for missing values
    if df_features.isnull().values.any():
        raise ValueError("Input data contains missing values. Please handle them beforeprediction.")
    
    # Scale the data using the fitted scaler
    scaled_data = scaler.transform(df_features)
    
    # Select the latest 'window_size' records
    if len(scaled_data) < window_size:
        raise ValueError(f"Insufficient data: requires at least {window_size} records, butgot {len(scaled_data)}.")
    
    latest_window = scaled_data[-window_size:]
    
    # Reshape to (1, window_size, num_features) for LSTM input
    reshaped_input = latest_window.reshape((1, window_size, len(features)))
    
    return reshaped_input

def postprocess_output(prediction, scaler, features=['Soil_Moisture', 'sin_day', 'cos_day', 'month']):
    """
    Converts the model's scaled prediction back to the original soil moisture scale.
    Parameters:
    - prediction (np.ndarray): The raw prediction output from the model (scaled).
    Typically of shape (1, 1).
    - scaler (MinMaxScaler): The scaler fitted on the training data.
    - features (list): List of feature column names used during scaling.
    Returns:
    - float: The soil moisture value in the original scale.
    """
    # Ensure the prediction is a NumPy array
    if not isinstance(prediction, np.ndarray):
        prediction = np.array(prediction)
    
    # Flatten prediction if necessary
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        prediction = prediction.flatten()
    elif prediction.ndim != 1:
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
    
    # Create a placeholder array with zeros for other features
    placeholder = np.zeros((len(prediction), len(features)))
    
    # Assign the predicted soil moisture values to the 'Soil_Moisture' column (assumed tobe the first feature)
    placeholder[:, 0] = prediction
    
    # Apply the inverse transformation to revert scaling
    inversed = scaler.inverse_transform(placeholder)
    
    # Extract the 'Soil_Moisture' values from the first column
    soil_moisture = inversed[:, 0]
    
    # Return as a single float value
    return float(soil_moisture[0])

def helper():
    df = pd.read_csv('./server/controllers/specific_region_data.csv', parse_dates=['Date'])
    
    df['Date'] = df['Date'].dt.date

    today_date = datetime.today().date()  
    print(f"Checking for date: {today_date}")

    if today_date in df['Date'].values:
        
        existing_soil_moisture = df.loc[df['Date'] == today_date, 'Soil_Moisture'].values[0]
        return existing_soil_moisture
    
    
    scaler = joblib.load('./server/controllers/scaler.joblib')

    model = load_model('./server/controllers/Soil_m_1.h5')

    try:
        processed_input = preprocess_input(
            new_data=df,
            scaler=scaler,
            window_size=195,
            features=['Soil_Moisture', 'sin_day', 'cos_day', 'month'],
            date_format='%Y-%m-%d'
        )
        
        raw_prediction = model.predict(processed_input)
        
        actual_soil_moisture = postprocess_output(
            prediction=raw_prediction,
            scaler=scaler,
            features=['Soil_Moisture', 'sin_day', 'cos_day', 'month']
        )
        
        print(f"Predicted Soil Moisture for the next day: {actual_soil_moisture:.4f}")
        
        csv_file_path = './server/controllers/specific_region_data.csv'
        update_csv(csv_file_path, actual_soil_moisture)
        
        return actual_soil_moisture
    
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
    

def predict_soil():
    try:

        value = helper()

        df = pd.read_csv('./server/controllers/specific_region_data.csv', parse_dates=['Date'])
        last_100_soil_moisture = df[['Soil_Moisture']].tail(20).to_dict(orient='records')
        
        response = make_response(jsonify({
            "success": True,
            "Predicted": value,
            "data": last_100_soil_moisture
        }), 201)
        
        return response
        
        return response
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred"}), 500



def prediction(date):
    try:
        
        with open('server/controllers/comp_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # predictions = model.forecast(steps=4)
        predictions = model.predict(date)
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise  

try:
    temp = prediction(date)
    print(f"Temp output from direct prediction: {temp}")
except Exception as e:
    print(f"Direct test error: {e}")

def predict_air_quality():
    try:
        date = request.json.get('date')
        if not date:
            return jsonify({"error": "Date is required"}), 400
        
        predictions = prediction(date)  
        
        predictions_dict = {str(k): v for k, v in predictions.items()}
        
        
        return jsonify({'response': predictions_dict})
    except Exception as e:
        print(f"Error in route: {str(e)}")
        return jsonify({"error": str(e)}), 400





# Load the Keras model (H5 model)
model = tf.keras.models.load_model('server/controllers/mobilenetv2_crop_disease.h5')

class_labels = ['Cassava Bacterial Blight (CBB)',
                'Cassava Brown Streak Disease (CBSD)',
                'Cassava Green Mottle (CGM)',
                'Cassava Mosaic Disease (CMD)',
                'Healthy']


def preprocess_image(image_file):
    
    img = Image.open(image_file).convert('L')
    
    img = img.resize((128, 128))
    
    img_array = np.array(img).astype(np.float32)
    
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

def predict_vegetation():
    try:
        image = request.files.get('fileup')

        preprocessed_image = preprocess_image(image)

        predictions = model.predict(preprocessed_image)

        predicted_class = np.argmax(predictions[0])

        class_name = class_labels[predicted_class]

        return jsonify({'predicted_class': class_name})

    except Exception as e:
        return jsonify({'error': str(e)})

