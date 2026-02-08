import torch
import joblib
import numpy as np
import pandas as pd
from collections import deque
import json
import os

from macro_detector.TransformerMacroDetector import TransformerMacroAutoencoder
from macro_detector.indicators import indicators_generation
from macro_detector.loss_caculation import Loss_Calculation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "assets", "mouse_macro_lstm_best.pt")
DEFAULT_SCALER_PATH = os.path.join(BASE_DIR, "assets", "scaler.pkl")

class MacroDetector:
    def __init__(self, config_path):
        
        self.cfg:dict = {}
        with open(config_path, 'r') as f:
            self.cfg:dict = json.load(f)

        self.seq_len = self.cfg.get("seq_len", 50)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.allowable_add_data = self.seq_len + 1 # 여유값 + 1
        self.CLIP_BOUNDS:dict = self.cfg["CLIP_BOUNDS"]
        self.features = list(self.CLIP_BOUNDS.keys())
        self.input_size = len(self.features)
        self.weight_threshold = self.cfg["weight_threshold"]

        self.base_threshold = self.cfg['threshold']
        self.buffer = deque(maxlen=self.allowable_add_data)
        
        # 노이즈 방지를 위해 최근 3~5개 에러의 평균만 사용 (순간적인 튐 방지)
        self.smooth_error_buf = deque(maxlen=5)

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
        self.scaler = joblib.load(DEFAULT_SCALER_PATH)

    def push(self, data: dict):
        self.buffer.append(
            (
                data.get('x'), 
                data.get('y'), 
                data.get('timestamp'), 
                data.get('deltatime')
            )
        )
        
        if len(self.buffer) < self.allowable_add_data:
            return None
        
        return self._infer()

    def _infer(self):

        df = pd.DataFrame(list(self.buffer), columns=["x", "y", "timestamp", "deltatime"])
        df = indicators_generation(df)

        df_features = df[self.features ].tail(self.seq_len).copy()
        
        if self.CLIP_BOUNDS:
            for col, b in self.CLIP_BOUNDS.items():
                if col in df_features.columns:
                    df_features[col] = df_features[col].clip(lower=b['min'], upper=b['max'])

        try:
            X_scaled = self.scaler.transform(df_features.values)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(X_tensor)
                recon_error = Loss_Calculation(outputs=output, batch=X_tensor).item()
        except Exception as e:
            print(f"❌ Inference Error: {e}")
            return None

        self.smooth_error_buf.append(recon_error)
        avg_error = np.mean(self.smooth_error_buf)
        final_threshold = float(self.base_threshold * self.weight_threshold)

        return {
            "raw_error": float(round(avg_error, 5)),
            "threshold": final_threshold,
            "is_macro": avg_error > final_threshold
        }