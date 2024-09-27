from flask import request, jsonify, make_response
import random
from dotenv import load_dotenv

load_dotenv()

# Predict Flood Function
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

# Predict Drought Function (similar structure as predict_flood)
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
                "random_number": random.randint(1, 100),  # Fixed randint call
            }
        }), 201)
        
        return response
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred"}), 500

# Predict Air Quality Function
def predict_air_quality():
    return jsonify({"success": True, "message": "Air quality predicted successfully!"})




import numpy as np
import tensorflow as tf
# from flask import Flask, request, jsonify
from PIL import Image
import io
import base64


# # Load the Keras model (H5 model)
model = tf.keras.models.load_model('server/controllers/mobilenetv2_crop_disease.h5')

# Define the class labels (replace with your actual classes)
class_labels = ['Cassava Bacterial Blight (CBB)', 'Cassava Brown Streak Disease (CBSD)', 'Cassava Green Mottle (CGM)', 'Cassava Mosaic Disease (CMD)', 'Healthy']


# Function to preprocess the image (resize and rescale)
def preprocess_image(image_file):
    # Load the image from the file
    img = Image.open(image_file).convert('L')  # Convert to grayscale (L mode)
    
    # Resize the image to the target size (128x128)
    img = img.resize((128, 128))
    
    # Convert the image to a numpy array
    img_array = np.array(img).astype(np.float32)
    
    # Rescale the image (the model was trained with rescale=1./255)
    img_array = img_array / 255.0
    
    # Add a batch dimension (model expects input shape (1, 128, 128, 1))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)  # To add the channel dimension for grayscale
    
    return img_array

# Route to handle vegetation disease prediction
def predict_vegetation():
    try:
        data = request.get_json()
         
        image_data = data['data']
         
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        image = io.BytesIO(image_bytes)  # Convert bytes to a file-like object

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make a prediction using the model
        predictions = model.predict(preprocessed_image)

        # Get the predicted class (index of the highest probability)
        predicted_class = np.argmax(predictions[0])

        # Get the class label
        class_name = class_labels[predicted_class]

        # Return the result as a JSON response
        return jsonify({'predicted_class': class_name})

    except Exception as e:
        return jsonify({'error': str(e)})

# # Predict Vegetation Function
# def predict_vegetation():
#     return jsonify({"success": True, "message": "Vegetation health predicted successfully!"})
