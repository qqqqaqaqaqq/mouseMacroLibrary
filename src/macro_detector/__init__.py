import os
from macro_detector.macro_dectector import MacroDetector
from macro_detector.MousePoint import MousePoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "assets", "config.json")

_detector = MacroDetector(config_path=CONFIG_PATH)

def get_macro_result(receive_data: dict):

    user_data:list[MousePoint] = []

    user_data.append(MousePoint(**receive_data))
    result = None
    for step in user_data:
        
        p_data = {
            'timestamp': step.timestamp,
            'x': step.x,
            'y': step.y,
            'deltatime': step.deltatime
        }
        
        result = _detector.push(p_data)

        return result
        