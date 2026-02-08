import os
from macro_detector.macro_dectector import MacroDetector
from macro_detector.MousePoint import MousePoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "assets", "config.json")

_detector = MacroDetector(config_path=CONFIG_PATH)

def get_macro_result(receive_data_list: list[dict]):
    try:
        all_data = []
        result = {}        
        if len(receive_data_list) < 51:
            return {
                "status": "1",
                "message": f"데이터가 부족합니다. 현재 {len(receive_data_list)}개 보냈습니다. 최소 51개 이상 넣어주세요.",
                "hint": {}
            }
        
        inferenc_result = None

        for data in receive_data_list:
            step = MousePoint(**data)
            p_data = {
                'timestamp': step.timestamp,
                'x': step.x,
                'y': step.y,
                'deltatime': step.deltatime
            }
            
            inferenc_result = _detector.push(p_data)

            if inferenc_result is not None:
                all_data.append(inferenc_result)


        result = {
            "status": "0",
            "data" : all_data
        }
        _detector.buffer.clear()

        return result
    except Exception as e:
        return {
            "status": "1",
            "message": f"데이터 형식 오류입니다. 해당 데이터 형식으로 전달 해주세요.",
            "hint": {
                "example": [
                    {
                        "timestamp": "2026-02-08T20:48:29",
                        "x": 100,
                        "y": 200,
                        "deltatime": 0.016
                    }
                ],
                "description": "위와 같은 형식의 객체를 리스트에 담아 최소 51개 이상 POST 요청으로 보내야 분석이 시작됩니다."
            }
        }     