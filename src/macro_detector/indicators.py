import pandas as pd
import numpy as np

def indicators_generation(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df = df_chunk.copy()

    for col in ['x', 'y', 'deltatime']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)

    dt = df["deltatime"]

    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dist"] = np.hypot(df["dx"], df["dy"])

    df["speed"] = df["dist"] / dt
    df["acc"] = df["speed"].diff() / dt
    df["jerk"] = df["acc"].diff() / dt

    df["theta"] = np.arctan2(df["dy"], df["dx"])
    unwrapped = np.unwrap(df["theta"].fillna(0).values)
    df["angular_speed"] = (pd.Series(unwrapped, index=df.index).diff() / dt)
    df["direction_change"] = df["theta"].diff().abs()

    df["micro_shake"] = (df["dx"].diff().abs() + df["dy"].diff().abs())

    # log
    df["log_speed"] = np.log1p(df["speed"])
    df["log_micro_shake"] = np.log1p(df["micro_shake"])

    df["log_acc"] = np.sign(df["acc"]) * np.log1p(np.abs(df["acc"]))
    df["log_jerk"] = np.sign(df["jerk"]) * np.log1p(np.abs(df["jerk"]))
    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
