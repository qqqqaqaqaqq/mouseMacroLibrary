import torch
import numpy as np
import pandas as pd
from collections import deque

from sklearn.preprocessing import RobustScaler
from QMacroDetector.indicators import indicators_generation

from QMacroDetector.make_sequence import make_seq
from QMacroDetector.make_gauss import make_gauss
from QMacroDetector.loss_caculation import Loss_Calculation

class MacroDetector:
    def __init__(self, cfg, model, scaler, FEATURES, device):

        self.cfg:dict = cfg

        self.seq_len = self.cfg.get("seq_len", 50)
        self.tolerance = self.cfg.get("tolerance", 0.02)
        self.chunk_size = self.cfg.get("chunk_size", 50)

        self.allowable_add_data = self.seq_len + self.chunk_size + 30

        self.FEATURES = [
            "speed",
            "acc",
            "jerk",
            "micro_shake",
            "curvature",
            "angle_vel",
            "energy_impact", 
            "jerk_diff"
        ]

        self.input_size = len(FEATURES) * 4
        self.weight_threshold = self.cfg["weight_threshold"]

        self.base_threshold = self.cfg['threshold']
        self.buffer = deque(maxlen=self.allowable_add_data)

        self.device = device
        self.model = model
        self.scaler:RobustScaler = scaler

    def push(self, data: dict):
        self.buffer.append((data.get('x'), data.get('y'), data.get('timestamp'), data.get('deltatime')))
    
        if len(self.buffer) < self.allowable_add_data:
            return None
        
        return self._infer()

    def _infer(self):
        df = pd.DataFrame(list(self.buffer), columns=["x", "y", "timestamp", "deltatime"])
        
        df = df[df["deltatime"] <= self.tolerance * 10].reset_index(drop=True)
        
        df = indicators_generation(df)

        df_filter_chunk = df[self.FEATURES].copy()
        
        chunks_scaled_array = self.scaler.transform(df_filter_chunk)
        
        chunks_scaled_df = pd.DataFrame(chunks_scaled_array, columns=self.FEATURES)
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