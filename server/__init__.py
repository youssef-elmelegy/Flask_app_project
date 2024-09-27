from flask import Flask

def create_app():
    app = Flask(__name__)
    
    app.url_map.strict_slashes = False  # Handle trailing slashes if necessary
    
    from .routes.weather import weather_bp
    app.register_blueprint(weather_bp, url_prefix='/')
    
    return app
