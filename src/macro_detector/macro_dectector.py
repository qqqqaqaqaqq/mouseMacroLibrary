import torch
import joblib
import numpy as np
import pandas as pd
from collections import deque
import os
import json

from sklearn.preprocessing import RobustScaler

from macro_detector.TransformerMacroDetector import TransformerMacroAutoencoder
from macro_detector.indicators import indicators_generation

from macro_detector.make_sequence import make_seq
from macro_detector.make_gauss import make_gauss
from macro_detector.loss_caculation import Loss_Calculation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "assets", "mouse_macro_lstm_best.pt")
DEFAULT_SCALER_PATH = os.path.join(BASE_DIR, "assets", "scaler.pkl")


FEATURES = [
    "micro_shake",
    "speed",
    "acc",
    "jerk"
]

class MacroDetector:
    def __init__(self, config_path):

        self.cfg:dict = {}
        with open(config_path, 'r') as f:
            self.cfg:dict = json.load(f)

        self.seq_len = self.cfg.get("seq_len", 50)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tolerance = self.cfg.get("tolerance", 0.02)
        self.chunk_size = self.cfg.get("chunk_size", 50)

        self.allowable_add_data = self.seq_len + self.chunk_size + 5

        self.input_size = len(FEATURES) * 6
        self.weight_threshold = self.cfg["weight_threshold"]

        self.base_threshold = self.cfg['threshold']
        self.buffer = deque(maxlen=self.allowable_add_data)

        self.buffer = deque(maxlen=self.allowable_add_data * 2)

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

    def push(self, data: dict):
        self.buffer.append((data.get('x'), data.get('y'), data.get('timestamp'), data.get('deltatime')))
    
        if len(self.buffer) < self.allowable_add_data:
            return None
        
        return self._infer()

    def _infer(self):
        df = pd.DataFrame(list(self.buffer), columns=["x", "y", "timestamp", "deltatime"])
        
        df = df[df["deltatime"] <= self.tolerance * 10].reset_index(drop=True)
        
        df = indicators_generation(df)

        df_filter_chunk = df[FEATURES].copy()
        
        chunks_scaled_array = self.scaler.transform(df_filter_chunk)
        
        chunks_scaled_df = pd.DataFrame(chunks_scaled_array, columns=FEATURES)
        chunks_scaled = make_gauss(data=chunks_scaled_df, chunk_size=self.chunk_size, chunk_stride=1, offset=10, train_mode=False)
        
        if len(chunks_scaled) < self.seq_len:
            return None
        
        final_input:np.array = make_seq(data=chunks_scaled, seq_len=self.seq_len, stride=1)

        
        last_seq = torch.tensor(final_input[-1], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        if last_seq.shape[1] < self.seq_len:
            return None

        with torch.no_grad():
            output = self.model(last_seq)

            sample_errors = Loss_Calculation(outputs=output, batch=last_seq).item()

            # 임계치 판정 logic
            is_human = sample_errors <= self.base_threshold
            
            if not is_human:
                if hasattr(self, 'log_queue'):
                    print(f"🚨 [DETECTION] Error: {sample_errors:.4f}")

        return {
            "is_human": is_human,
            "macro_probability": "🚨 MACRO" if not is_human else "🙂 HUMAN",
            "prob_value": sample_errors, # score 대신 error 값 전달
            "raw_error": round(sample_errors, 5),
            "threshold": self.base_threshold
        }