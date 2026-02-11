import pandas as pd
import numpy as np
import sys

def make_gauss(data: pd.DataFrame, chunk_size: int, chunk_stride: int, offset: int, train_mode:bool=True) -> np.array:
    data_np = data.values[offset:] 
    chunks = []
    eps = 1e-9

    # 루프 범위 미리 계산
    loop_range = range(0, len(data_np) - chunk_size + 1, chunk_stride)
    total_steps = len(loop_range)

    for idx, i in enumerate(loop_range):
        window = data_np[i : i + chunk_size]
        
        # 1. 기존 통계량 (Moment)
        m = np.mean(window, axis=0)
        s = np.std(window, axis=0)
        diff = window - m
        sk = np.mean(diff**3, axis=0) / (s**3 + eps)
        kt = np.mean(diff**4, axis=0) / (s**4 + eps) - 3
        
        # 2. 선형성 및 연속성 지표
        diff_1 = np.diff(window, axis=0)
        roughness = np.mean(np.abs(diff_1), axis=0)
        
        # 3. 자기상관
        if chunk_size > 1:
            ac = np.mean(window[1:] * window[:-1], axis=0) / (np.var(window, axis=0) + eps)
        else:
            ac = np.zeros(window.shape[1])

        chunks.append(np.concatenate([m, s, sk, kt, roughness, ac]))

        if train_mode:
            # --- 네모박스 진행바 로직 ---
            if (idx + 1) % max(1, (total_steps // 50)) == 0 or (idx + 1) == total_steps:
                progress = (idx + 1) / total_steps
                bar_length = 20  # 전체 바 길이
                filled_length = int(bar_length * progress)
                # ㅁ는 완료, o는 미완료
                bar = '■' * filled_length + '□' * (bar_length - filled_length)
                
                # \r을 사용하여 한 줄에서 계속 업데이트
                sys.stdout.write(f'\r진행중: [{bar}] {progress*100:>5.1f}% ({idx+1}/{total_steps})')
                sys.stdout.flush()

    return np.array(chunks)