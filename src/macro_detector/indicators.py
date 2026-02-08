import pandas as pd
import numpy as np

def indicators_generation(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df = df_chunk.copy()
    
    # 0. 시간 안전장치 (dt가 너무 작으면 속도가 폭발함)
    dt = df["deltatime"].replace(0, 0.0001)
    
    df["dt_diff"] = df["deltatime"].diff().abs()
    df["dt_cv"] = (
        df["deltatime"].rolling(10, min_periods=1).std() /
        (df["deltatime"].rolling(10, min_periods=1).mean() + 1e-6)
    )

    # 1. 기본 물리량 계산 (이동 거리, 속도, 가속도, 저크)
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dist"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    
    df["speed"] = df["dist"] / dt
    df["acc"] = df["speed"].diff()
    df["jerk"] = df["acc"].diff()

    # 2. [추가] 미세 떨림 (Micro-Jitter) 분석
    # 사람이 이동 중 발생하는 불규칙한 미세 움직임 포착
    df["jitter_x"] = df["dx"].diff().abs()
    df["jitter_y"] = df["dy"].diff().abs()
    df["micro_shaking"] = (df["jitter_x"] + df["jitter_y"]).rolling(5, min_periods=1).mean()


    # 3. 가속도 방향 전환 빈도 (Jerk Flip)
    df["jerk_sign"] = np.sign(df["jerk"])
    df["jerk_flip"] = (df["jerk_sign"].diff().abs() > 1).astype(int)
    df["jerk_flip_rate"] = df["jerk_flip"].rolling(10, min_periods=1).mean()

    # 4. 방향 및 회전 관련 (Turn, Angular Velocity)
    df["angle"] = np.arctan2(df["dy"], df["dx"])
    df["turn"] = (df["angle"].diff() + np.pi) % (2 * np.pi) - np.pi
    
    # [추가] 각속도 및 각가속도 (곡선의 부드러움 측정)
    df["ang_vel"] = df["turn"] / dt
    df["ang_acc"] = df["ang_vel"].diff()

    # 5. 통계 피처 (분산 및 부드러움)
    df["speed_var"] = df["speed"].rolling(window=10, min_periods=1).std()
    
    # [추가] 가속도의 일정함 (매크로는 가속도 변화가 매우 규칙적임)
    df["acc_smoothness"] = df["acc"].abs() / (df["acc"].rolling(10, min_periods=1).std() + 1e-6)

    
    df["dx_dt"] = df["x"].diff() / dt
    df["dy_dt"] = df["y"].diff() / dt
    df["d2x_dt2"] = df["dx_dt"].diff() / dt
    df["d2y_dt2"] = df["dy_dt"].diff() / dt
    
    # 3. 곡률(Curvature) 계산
    # 수식: |x'y'' - y'x''| / (x'^2 + y'^2)^(1.5)
    num = (df["dx_dt"] * df["d2y_dt2"] - df["dy_dt"] * df["d2x_dt2"]).abs()
    # (x'^2 + y'^2)은 속도의 제곱임
    den = (df["dx_dt"]**2 + df["dy_dt"]**2)**(1.5)
    
    # 0으로 나누기 방지 (den이 0이면 NaN 발생 후 마지막에 fillna 처리)
    df["curvature"] = num / den

    # 4. 곡률 반경(R) 계산
    df["radius"] = 1.0 / df["curvature"]

    # 5. [핵심] 곡률 반경의 상대 변화율 계산
    # dR/dt 계산
    df["dr_dt"] = df["radius"].diff() / dt
    # 상대 변화율 ( (dR/dt) / R )
    df["radius_roc_rel"] = df["dr_dt"] / df["radius"]

    # 8. NaN/inf 최종 정리
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df