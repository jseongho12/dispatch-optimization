from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
import requests
import pickle
import config as cfg


def load_hubs(path: Path = cfg.HUB_PATH):
    df = pd.read_excel(path)
    return df[["lng", "lat"]].values.tolist()


def build_matrix(coords):
    coord_str = ";".join(f"{lng},{lat}" for lng, lat in coords)
    url = f"{cfg.OSRM_URL}/table/v1/driving/{coord_str}"
    params = {"annotations": "duration,distance"}

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if data.get("durations") is None:
        raise RuntimeError("OSRM returned null durations")

    time_matrix = np.array(data["durations"]) / 60.0     # minutes
    dist_matrix = np.array(data["distances"]) / 1000.0   # km

    return time_matrix, dist_matrix


if __name__ == "__main__":
    coords = load_hubs()
    time_m, dist_m = build_matrix(coords)

    with open(cfg.DATA_PATH / "time_matrix.pkl", "wb") as f:
        pickle.dump(time_m, f)

    with open(cfg.DATA_PATH / "dist_matrix.pkl", "wb") as f:
        pickle.dump(dist_m, f)

    print("✅ time_matrix (minutes), dist_matrix (km) saved.")