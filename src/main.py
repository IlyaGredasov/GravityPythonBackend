import os

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from handlers import register_handlers
from routes import register_routes

app = Flask(__name__)
CORS(app)
load_dotenv('.env')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

register_handlers(socketio)
register_routes(app, socketio)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
