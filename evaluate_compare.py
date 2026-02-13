# evaluate_compare.py
import os
import json
import random
import argparse
import pickle
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
from simulator import Simulator
from solver import solve_pdptw  # solver.py에 있음 :contentReference[oaicite:2]{index=2}


HORIZON = 24 * 60


# ---------------------------
# KPI 계산 (solver.log_print의 합계 로직을 "출력 없이" 재구현)
# ---------------------------
def compute_kpis(res, df_all, time_matrix, dist_matrix, horizon=HORIZON):
    """
    반환:
      dispatch_rate, total_km, total_time_min, travel_min, wait_min, wait_rate,
      used_vehicles, dropped
    """
    if res is None:
        return {
            "dispatch_rate": 0.0,
            "total_km": None,
            "total_time_min": None,
            "travel_min": None,
            "wait_min": None,
            "wait_rate": None,
            "used_vehicles": 0,
            "dropped": None,
        }

    df_all = df_all.copy()
    if "ORDER_ID" not in df_all.columns:
        df_all["ORDER_ID"] = range(1, len(df_all) + 1)

    order_map = df_all.set_index("ORDER_ID")

    total_travel_min = 0
    total_wait_min = 0
    total_km = 0.0
    total_done_orders = 0

    paid_orders = set()

    for v, route in enumerate(res["routes"]):
        # 업무 있는 차량만
        has_job = any(step["action"] in ("PU", "DO") and step.get("active", 1) == 1 for step in route)
        if not has_job:
            continue

        prev_start = None
        prev_hub = None
        first_pu_done = False

        for step in route:
            action = step["action"]
            hub = int(step["hub"])
            start = int(step["arr_min"])

            # 이동 구간
            if prev_start is None:
                seg_travel = 0
                seg_km = 0.0
                eta = start
            else:
                seg_travel = int(time_matrix[prev_hub][hub])
                seg_km = float(dist_matrix[prev_hub][hub])
                eta = prev_start + seg_travel

            # TW 기반 대기 계산 (solver.log_print 방식과 동일한 스타일)
            if action in ("DEPOT", "END"):
                waiting = 0
                oid = None
            else:
                oid = int(step["order_id"])
                row = order_map.loc[oid]
                tw_start = int(row["PU_TIME_IDX"])  # solver.log_print는 PU/DO 둘 다 PU_TIME_IDX를 쓰는 방식 :contentReference[oaicite:3]{index=3}
                waiting = max(0, tw_start - eta)

                # 첫 PU는 강제로 TW에 맞춰찍는 처리(대기 0)
                if (not first_pu_done) and action == "PU":
                    waiting = 0
                    first_pu_done = True

            total_travel_min += seg_travel
            total_wait_min += waiting
            total_km += seg_km

            # 완료 주문 수는 DO에서 카운트(중복 방지)
            if action == "DO" and oid is not None and oid not in paid_orders:
                paid_orders.add(oid)
                total_done_orders += 1

            prev_start = start
            prev_hub = hub

    n_total_orders = len(df_all)
    total_time_with_wait = total_travel_min + total_wait_min
    dispatch_rate = (total_done_orders / n_total_orders) if n_total_orders else 0.0
    wait_rate = (total_wait_min / total_time_with_wait) if total_time_with_wait else 0.0

    dropped = len(res.get("unassigned_orders", []))
    used_vehicles = int(res.get("used_vehicles", 0))

    return {
        "dispatch_rate": float(dispatch_rate),
        "total_km": float(total_km),
        "total_time_min": int(total_time_with_wait),
        "travel_min": int(total_travel_min),
        "wait_min": int(total_wait_min),
        "wait_rate": float(wait_rate),
        "used_vehicles": used_vehicles,
        "dropped": dropped,
    }


# ---------------------------
# Plot helpers (matplotlib only)
# ---------------------------
def boxplot_two_versions(df: pd.DataFrame, metric: str, out_png: str):
    versions = ["before", "after"]
    data = [df[df["version"] == v][metric].dropna().values for v in versions]

    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=versions, showmeans=True)
    plt.title(metric)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def bar_mean(df: pd.DataFrame, metric: str, out_png: str):
    m = df.groupby("version")[metric].mean()

    plt.figure(figsize=(6, 4))
    plt.bar(m.index.tolist(), m.values.tolist())
    plt.title(f"Mean: {metric}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50, help="number of simulations")
    ap.add_argument("--seed", type=int, default=42, help="base random seed")
    ap.add_argument("--outdir", type=str, default="compare_outputs", help="output folder")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) matrices load (solver.py main과 동일) :contentReference[oaicite:4]{index=4}
    with open(os.path.join(cfg.DATA_PATH, "time_matrix.pkl"), "rb") as f:
        time_matrix = pickle.load(f)
    with open(os.path.join(cfg.DATA_PATH, "dist_matrix.pkl"), "rb") as f:
        dist_matrix = pickle.load(f)

    trucks_path = str(cfg.TRUCKS_PATH)  # config.py에 정의 :contentReference[oaicite:5]{index=5}

    # 2) before/after params
    BEFORE =  {
        'time_limit_sec': 10,
        'drop_penalty': 10_000_000,   
        'frozen_dry_penalty': 10_000_000,   
        'vehicle_fixed_cost': 240_000,      
        'wait_cost_coeff': 200,             
    }
    

    AFTER = dict(cfg.config["params"])  # 현재 config.py의 optuna 결과 :contentReference[oaicite:6]{index=6}

    sim = Simulator()  # simulator.py :contentReference[oaicite:7]{index=7}

    records = []
    for r in range(args.runs):
        # ✅ 같은 df로 before/after 공정 비교: run별 seed 고정
        random.seed(args.seed + r)
        df = sim.generate_orders(save=False)

        for version, P in [("before", BEFORE), ("after", AFTER)]:
            res = solve_pdptw(
                df=df,
                time_matrix=time_matrix,
                dist_matrix=dist_matrix,
                trucks_path=trucks_path,
                depot_hub=0,

                allow_drop=True,
                drop_penalty=int(P["drop_penalty"]),
                vehicle_fixed_cost=int(P["vehicle_fixed_cost"]),
                wait_cost_coeff=int(P["wait_cost_coeff"]),
                frozen_dry_penalty=int(P["frozen_dry_penalty"]),
                time_limit_sec=int(P["time_limit_sec"]),

                objective="dist",
                capacity_scale=100,
            )

            k = compute_kpis(res, df, time_matrix, dist_matrix)

            records.append({
                "run_id": r,
                "version": version,
                **P,
                **k,
                "objective": None if res is None else int(res.get("objective", -1)),
            })

        print(f"[{r+1:03d}/{args.runs}] done")

    df_res = pd.DataFrame(records)

    # 3) save CSV + params snapshot
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.outdir, f"compare_{ts}.csv")
    df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(os.path.join(args.outdir, f"params_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({"before": BEFORE, "after": AFTER}, f, ensure_ascii=False, indent=2)

    # 4) plots
    metrics = [
        "dispatch_rate",
        "total_km",
        "total_time_min",
        "wait_rate",
        "used_vehicles",
        "dropped",
    ]

    for m in metrics:
        boxplot_two_versions(df_res, m, os.path.join(args.outdir, f"box_{m}_{ts}.png"))
        bar_mean(df_res, m, os.path.join(args.outdir, f"mean_{m}_{ts}.png"))

    # 5) quick summary
    summary = df_res.groupby("version")[metrics].agg(["mean", "std"])
    print("\n=== SUMMARY (mean/std) ===")
    print(summary)
    print(f"\nSaved: {csv_path}")
    print(f"Plots in: {args.outdir}")


if __name__ == "__main__":
    main()
