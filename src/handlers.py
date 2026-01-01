from flask import request
from flask_socketio import SocketIO

from routes import pools_dict
from routes import stop_execution_pool


def register_handlers(socketio: SocketIO):
    @socketio.on('disconnect')
    def handle_disconnect():
        stop_execution_pool(request.sid)

    @socketio.on('button_press')
    def handle_button_press(data):
        user_id = request.sid
        if user_id in pools_dict.keys():
            simulation = pools_dict[user_id].simulation
            match data['direction']:
                case 'right':
                    simulation.controllable_acceleration.right = data['is_pressed']
                case 'left':
                    simulation.controllable_acceleration.left = data['is_pressed']
                case 'up':
                    simulation.controllable_acceleration.up = data['is_pressed']
                case 'down':
                    simulation.controllable_acceleration.down = data['is_pressed']
                case _:
                    raise ValueError("Invalid direction")
