from flask import Blueprint
from server.controllers.weather_controller import (
    predict_flood, 
    predict_drought, 
    predict_air_quality, 
    predict_vegetation,
    predict_soil
)

weather_bp = Blueprint('weather', __name__)

# Routes for prediction endpoints
weather_bp.route('/predict-flood', methods=['POST'])(predict_flood)
weather_bp.route('/predict-drought', methods=['POST'])(predict_drought)
weather_bp.route('/predict-soil', methods=['POST'])(predict_soil)
weather_bp.route('/predict-air-quality', methods=['POST'])(predict_air_quality)
weather_bp.route('/predict-vegetation', methods=['POST'])(predict_vegetation)

