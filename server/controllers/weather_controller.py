from flask import request, jsonify, make_response
import random
from dotenv import load_dotenv

import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

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

def predict_drought():
    try:
        data = request.get_json()
        name = data.get('name')
        input_data = data.get('data')

        if not name or not input_data:
            raise ValueError("All fields are required")

        response = make_response(jsonify({
            "success": True,
            "message": "Model predicted successfully",
            "response": {
                "name": name,
                "data": input_data,
                "random_number": random.randint(1, 100),  
            }
        }), 201)
        
        return response
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred"}), 500


import pickle


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

