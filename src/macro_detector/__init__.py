import os
from macro_detector.macro_dectector import MacroDetector
from macro_detector.MousePoint import MousePoint
from typing import List

import torch
import joblib
import json

from sklearn.preprocessing import RobustScaler
from macro_detector.TransformerMacroDetector import TransformerMacroAutoencoder

class Circle_Trajectory:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CONFIG_PATH = os.path.join(BASE_DIR, "assets", "security_circle_trajectory_model", "config.json")
        DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "assets", "security_circle_trajectory_model", "model.pt")
        DEFAULT_SCALER_PATH = os.path.join(BASE_DIR, "assets", "security_circle_trajectory_model", "scaler.pkl")

        self.cfg:dict = {}
        with open(CONFIG_PATH, 'r') as f:
            self.cfg:dict = json.load(f)

        FEATURES = [
            "speed",
            "acc",
            "jerk",
            "micro_shake",
            "curvature",
            "angle_vel",
            "energy_impact", 
            "jerk_diff"
        ]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = len(FEATURES) * 4
        # ===== 모델 초기화 =====
        self.model = TransformerMacroAutoencoder(
            input_size=self.input_size,
            d_model=self.cfg["d_model"],
            nhead=self.cfg["n_head"],
            num_layers=self.cfg["num_layers"],
            dim_feedforward=self.cfg["dim_feedforward"],
            dropout=self.cfg["dropout"]
        ).to(self.device)

        self.model.load_state_dict(torch.load(DEFAULT_MODEL_PATH, map_location=self.device, weights_only=True))
        self.model.eval()
        self.scaler:RobustScaler = joblib.load(DEFAULT_SCALER_PATH)

        self.detector = MacroDetector(cfg=CONFIG_PATH, model=self.model, scaler=self.scaler, FEATURES=FEATURES, device=self.device)        

    def get_macro_result(self, receive_data_list: List[MousePoint]):
        print(f"version 0.0.6")
        print(f"송신받은 데이터 개수 {len(receive_data_list)}")
        try:
            all_data = []
            result = {}        
            if len(receive_data_list) < self.detector.allowable_add_data:
                return {
                    "status": "1",
                    "message": f"데이터가 부족합니다. 현재 {len(receive_data_list)}개 보냈습니다. 최소 51개 이상 넣어주세요.",
                    "hint": {}
                }
            
            inferenc_result = None

            for data in receive_data_list:

                p_data = {
                    'timestamp': data.timestamp,
                    'x': data.x,
                    'y': data.y,
                    'deltatime': data.deltatime
                }
                
                inferenc_result = self.detector.push(p_data)

                if inferenc_result is not None:
                    all_data.append(inferenc_result)


            result = {
                "status": "0",
                "data" : all_data
            }
            self.detector.buffer.clear()

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