import os
import json
import random
import numpy as np
import pandas as pd

from typing import Dict, Any, Tuple, Optional, List  # ← 이거 추가

from solver import solve_pdptw
from simulator import Simulator
import config as cfg


def generate_one_scenario(seed: int):
    """
    Simulator 기반으로:
    df_orders, time_matrix, dist_matrix, trucks_path 반환
    """

    random.seed(seed)
    np.random.seed(seed)

    # -------------------------
    # simulator 실행
    # -------------------------
    sim = Simulator()
    df_orders = sim.generate_orders(save=False)  # dataframe 반환 :contentReference[oaicite:1]{index=1}

    # -------------------------
    # matrix
    # -------------------------
    time_matrix = sim.time_matrix
    dist_matrix = sim.dist_matrix

    # -------------------------
    # trucks 파일 경로
    # -------------------------
    trucks_path = os.path.join(cfg.DATA_PATH, "trucks.xlsx")

    return df_orders, time_matrix, dist_matrix, trucks_path


# -----------------------------
# 2) Feature engineering
# -----------------------------
def build_order_features(df: pd.DataFrame, time_mx, dist_mx) -> pd.DataFrame:
    df = df.copy()

    # 필수 컬럼 체크
    req = ["PU_HUB", "DO_HUB", "CAPACITY", "PU_TIME_IDX", "DO_TIME_IDX", "DELIVERY_TYPE"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "ORDER_ID" not in df.columns:
        df["ORDER_ID"] = np.arange(1, len(df) + 1)

    # Frozen 여부
    df["is_frozen_order"] = (df["DELIVERY_TYPE"].astype(str).str.strip() == "Frozen").astype(int)

    # 시간 특징
    df["tw_slack_min"] = (df["DO_TIME_IDX"].astype(int) - df["PU_TIME_IDX"].astype(int)).clip(lower=0)
    df["pu_hour"] = (df["PU_TIME_IDX"].astype(int) // 60).clip(0, 23)
    df["do_hour"] = (df["DO_TIME_IDX"].astype(int) // 60).clip(0, 23)

    # 직행 거리/시간 (PU->DO)
    pu = df["PU_HUB"].astype(int).to_numpy()
    do = df["DO_HUB"].astype(int).to_numpy()
    direct_time = np.array([int(time_mx[a][b]) for a, b in zip(pu, do)])
    direct_dist = np.array([float(dist_mx[a][b]) for a, b in zip(pu, do)])

    df["direct_time"] = direct_time
    df["direct_dist"] = direct_dist

    # 용량 로그/스케일
    df["cap"] = df["CAPACITY"].astype(float)
    df["cap_log1p"] = np.log1p(df["cap"])

    # 학습용 feature만 남기기
    feat_cols = [
        "ORDER_ID",
        "PU_HUB", "DO_HUB",
        "cap", "cap_log1p",
        "PU_TIME_IDX", "DO_TIME_IDX",
        "tw_slack_min",
        "pu_hour", "do_hour",
        "direct_time", "direct_dist",
        "is_frozen_order",
    ]
    return df[feat_cols]


# -----------------------------
# 3) Labels
# -----------------------------
def labels_from_solution(df_orders: pd.DataFrame, sol: Dict[str, Any]) -> pd.DataFrame:
    df = df_orders.copy()
    if "ORDER_ID" not in df.columns:
        df["ORDER_ID"] = np.arange(1, len(df) + 1)

    unassigned = set(sol.get("unassigned_orders", []))
    # solver는 unassigned_orders를 "주문 index(i)"로 반환 :contentReference[oaicite:2]{index=2}
    y_assigned = [(0 if i in unassigned else 1) for i in range(len(df))]
    out = pd.DataFrame({
        "ORDER_ID": df["ORDER_ID"].astype(int).to_numpy(),
        "y_assigned": np.array(y_assigned, dtype=int),
        "objective_full": int(sol.get("objective", -1)),
    })
    return out


def compute_delta_objective_per_order(
    df_orders: pd.DataFrame,
    time_mx,
    dist_mx,
    trucks_path: str,
    solver_kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """
    회귀 라벨: Δobjective_i = objective_full - objective_without_i
    - 비용: 주문 수 * solver 1회 추가 (느릴 수 있음)
    """
    # full
    sol_full = solve_pdptw(df_orders, time_mx, dist_mx, trucks_path=trucks_path, **solver_kwargs)
    if sol_full is None:
        raise RuntimeError("Full solve failed (no feasible). Try allow_drop=True / bigger time_limit_sec.")

    obj_full = int(sol_full["objective"])

    deltas = []
    for i in range(len(df_orders)):
        df_wo = df_orders.drop(df_orders.index[i]).reset_index(drop=True)
        sol_wo = solve_pdptw(df_wo, time_mx, dist_mx, trucks_path=trucks_path, **solver_kwargs)
        if sol_wo is None:
            # 제거했는데도 infeasible이면 이상하지만, 안전 처리
            deltas.append(np.nan)
            continue
        obj_wo = int(sol_wo["objective"])
        deltas.append(obj_full - obj_wo)

    if "ORDER_ID" not in df_orders.columns:
        order_ids = np.arange(1, len(df_orders) + 1)
    else:
        order_ids = df_orders["ORDER_ID"].astype(int).to_numpy()

    return pd.DataFrame({"ORDER_ID": order_ids, "y_delta_obj": np.array(deltas, dtype=float)})


# -----------------------------
# 4) Main: dataset generation
# -----------------------------
def main(
    out_dir: str = "ml_data",
    n_scenarios: int = 1000,
    seed0: int = 20260213,
    make_regression_labels: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    solver_kwargs = dict(
        depot_hub=0,
        horizon=24 * 60,
        time_limit_sec=10,        # 데이터 생성이라 타임 늘릴수록 라벨 품질↑
        allow_drop=True,
        drop_penalty=10_000_000,
        objective="dist",        # 너가 dist로 바꾸면 라벨도 그 objective 기준
        capacity_scale=100,      # solver 기본과 맞추기(소수용량 보존) :contentReference[oaicite:3]{index=3}
    )
    

    rows = []
    for s in range(n_scenarios):
        seed = seed0 + s
        df_orders, time_mx, dist_mx, trucks_path = generate_one_scenario(seed)

        sol = solve_pdptw(df_orders, time_mx, dist_mx, trucks_path=trucks_path, **solver_kwargs)
        if sol is None:
            print(f"[SCN {s:03d}] no-solution -> skip")
            continue

        X = build_order_features(df_orders, time_mx, dist_mx)
        y = labels_from_solution(df_orders, sol)
        df_xy = X.merge(y, on="ORDER_ID", how="inner")

        if make_regression_labels:
            df_delta = compute_delta_objective_per_order(
                df_orders, time_mx, dist_mx, trucks_path, solver_kwargs
            )
            df_xy = df_xy.merge(df_delta, on="ORDER_ID", how="left")

        df_xy["scenario_id"] = s
        df_xy["seed"] = seed

        rows.append(df_xy)

        print(f"[SCN {s:03d}] orders={len(df_orders)} obj={sol['objective']} assigned_rate={df_xy['y_assigned'].mean():.3f}")

    if not rows:
        raise RuntimeError("No scenarios produced any solution. Check generate_one_scenario() adapter.")

    data = pd.concat(rows, ignore_index=True)

    out_path = os.path.join(out_dir, "dataset.csv")
    data.to_csv(out_path, index=False, encoding="utf-8-sig")

    meta = {
        "n_rows": int(len(data)),
        "n_scenarios": int(data["scenario_id"].nunique()),
        "make_regression_labels": bool(make_regression_labels),
        "solver_kwargs": solver_kwargs,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved: {out_path}")
    print(f"✅ Meta : {os.path.join(out_dir, 'meta.json')}")


if __name__ == "__main__":
    main(make_regression_labels=True)

