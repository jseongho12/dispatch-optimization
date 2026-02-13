# tune_params.py
"""
Optuna tuning for OR-Tools PDPTW parameters with:
- Policy costs fixed inside solver (drop_penalty, frozen_dry_penalty)
- Quality gate on dispatch_rate (>= 90%)
- Optimization priority among PASS trials:
    1) used vehicles (min)
    2) total distance (min)
    3) total waiting (min)
  and frozen-dry is soft (discouraged but allowed)

Why gate?
- If you ignore drop/frozen in evaluation, Optuna may pick params that look cheap
  but violate service quality.
- If you put drop as top lexicographic term, Optuna mostly optimizes drop only.
- Gate gives the best of both: guarantee minimum service, then optimize costs.
"""

import os
import json
import pickle
import random
import statistics
from datetime import datetime

import optuna
import numpy as np

import config as cfg
from simulator import Simulator
from solver import solve_pdptw


# -----------------------------
# Fixed runtime settings
# -----------------------------
TIME_LIMIT_SEC = int(cfg.config["params"]["time_limit_sec"])
OBJECTIVE = "dist"
ALLOW_DROP = True  # keep allow_drop True; gate handles service threshold

# Policy (fixed inside solver)
DROP_PENALTY = int(cfg.config["params"]["drop_penalty"])
FROZEN_DRY_PENALTY = int(cfg.config["params"]["frozen_dry_penalty"])

# Fair tuning: fixed order count per episode
ORDER_N = 50

# Episodes per trial
EPISODES_PER_TRIAL = 20

# Optuna trials
N_TRIALS = 80

# Reproducibility
BASE_SEED = 42
EVAL_SEEDS = [BASE_SEED + i for i in range(EPISODES_PER_TRIAL)]

# Logging
KPI_LOG_PATH = "optuna_kpi_log.jsonl"


# -----------------------------
# Gate / thresholds (your policy)
# -----------------------------
MIN_DISPATCH_RATE = 0.90              # >= 90% OK
MAX_DROP_RATIO = 1.0 - MIN_DISPATCH_RATE  # 0.10
# Frozen-dry is allowed (not absolute forbidden).
# Optional: if you still want a soft cap, set it here.
SOFT_MAX_FROZEN_DRY = None  # e.g. 5, or None to disable soft-cap gate


# -----------------------------
# Global caches
# -----------------------------
_SIM = None
_TIME_MATRIX = None
_DIST_MATRIX = None


def load_matrices():
    with open(os.path.join(cfg.DATA_PATH, "time_matrix.pkl"), "rb") as f:
        time_matrix = pickle.load(f)
    with open(os.path.join(cfg.DATA_PATH, "dist_matrix.pkl"), "rb") as f:
        dist_matrix = pickle.load(f)
    return time_matrix, dist_matrix


def init_globals():
    global _SIM, _TIME_MATRIX, _DIST_MATRIX
    if _SIM is None:
        _SIM = Simulator()
    if _TIME_MATRIX is None or _DIST_MATRIX is None:
        _TIME_MATRIX, _DIST_MATRIX = load_matrices()


# -----------------------------
# KPI helpers (from solver outputs)
# -----------------------------
def _compute_total_km(res, dist_matrix) -> float:
    """Sum km from route steps (active only)."""
    total_km = 0.0
    for route in res.get("routes", []):
        prev = None
        for step in route:
            if int(step.get("active", 1)) != 1:
                continue
            hub = int(step["hub"])
            if prev is not None:
                total_km += float(dist_matrix[prev][hub])
            prev = hub
    return float(total_km)


def _compute_wait_min(res) -> int:
    """Wait = max(0, TW_start - arrival) for PU/DO steps (active only)."""
    total_wait = 0
    for route in res.get("routes", []):
        for step in route:
            if int(step.get("active", 1)) != 1:
                continue
            if step.get("action") not in ("PU", "DO"):
                continue
            arr = int(step.get("arr_min", 0))
            tw_s = int(step.get("tw_start_min", 0))
            total_wait += max(0, tw_s - arr)
    return int(total_wait)


def _compute_frozen_dry_count(res, df_orders) -> int:
    """
    Count cases where a Frozen vehicle served a non-Frozen order (PU steps).
    NOTE: this relies on res['frozen_vehicle_indices'] being provided by solver.
    """
    frozen_set = set(res.get("frozen_vehicle_indices", []))
    if not frozen_set:
        return 0

    if "ORDER_ID" not in df_orders.columns or "DELIVERY_TYPE" not in df_orders.columns:
        return 0
    dt_map = df_orders.set_index("ORDER_ID")["DELIVERY_TYPE"].to_dict()

    cnt = 0
    for v, route in enumerate(res.get("routes", [])):
        if v not in frozen_set:
            continue
        for step in route:
            if int(step.get("active", 1)) != 1:
                continue
            if step.get("action") != "PU":
                continue
            oid = step.get("order_id")
            if oid is None:
                continue
            if str(dt_map.get(int(oid), "")).strip() != "Frozen":
                cnt += 1
    return int(cnt)


def gate_ok(drop_cnt: int, order_n: int, frozen_dry_cnt: int) -> bool:
    """
    Gate condition:
    - dispatch_rate >= MIN_DISPATCH_RATE  <=> drop_cnt <= floor(order_n * MAX_DROP_RATIO)
    - optional soft cap on frozen-dry count (disabled by default)
    """
    max_drop = int(np.floor(order_n * MAX_DROP_RATIO))
    if drop_cnt > max_drop:
        return False

    if SOFT_MAX_FROZEN_DRY is not None and frozen_dry_cnt > int(SOFT_MAX_FROZEN_DRY):
        return False

    return True


def evaluate_with_gate(res, df_orders, dist_matrix, order_n: int):
    """
    Return (score, metrics) with:
    - FAIL: huge penalty (so Optuna learns to satisfy gate)
    - PASS: optimize used -> dist -> wait, with soft frozen penalty (small compared to vehicle)
    """
    # If solver returned nothing
    if res is None:
        big = 1e30
        return float(big), {
            "pass": False,
            "drop": None,
            "dispatch_rate": None,
            "frozen_dry": None,
            "used": None,
            "km": None,
            "dist_m": None,
            "wait_min": None,
        }

    drop = int(len(res.get("unassigned_orders", [])))
    used = int(res.get("used_vehicles", 0))
    km = _compute_total_km(res, dist_matrix)
    dist_m = float(km * 1000.0)
    wait_min = int(_compute_wait_min(res))
    frozen_dry = int(_compute_frozen_dry_count(res, df_orders))

    dispatch_rate = 1.0 - (drop / float(order_n)) if order_n > 0 else 0.0

    passed = gate_ok(drop, order_n, frozen_dry)

    # --- FAIL score (gate violation) ---
    # Huge base + how much violated (gives gradient to search)
    if not passed:
        max_drop = int(np.floor(order_n * MAX_DROP_RATIO))
        excess_drop = max(0, drop - max_drop)
        excess_frozen = 0
        if SOFT_MAX_FROZEN_DRY is not None:
            excess_frozen = max(0, frozen_dry - int(SOFT_MAX_FROZEN_DRY))

        # Make gate failures dominate everything.
        # Still add tiny tie-breakers so "less bad" failures are preferred.
        score = (
            1e30
            + excess_drop * 1e15
            + excess_frozen * 1e13
            + used * 1e9
            + dist_m * 1e3
        )
        metrics = {
            "pass": False,
            "drop": drop,
            "dispatch_rate": float(dispatch_rate),
            "frozen_dry": frozen_dry,
            "used": used,
            "km": float(km),
            "dist_m": float(dist_m),
            "wait_min": wait_min,
        }
        return float(score), metrics

    # --- PASS score (your priority) ---
    # Priority weights:
    # - used dominates (1st)
    # - dist dominates wait (2nd)
    # - wait is last
    #
    # Frozen-dry is "discouraged but allowed":
    # We penalize it roughly like +200km per violation (tunable).
    # Since dist term uses dist_m*1e3, 1km contributes 1e6 to score.
    # So 200km => 200 * 1e6 = 2e8.
    FROZEN_SOFT_PENALTY = 2e8  # ~200km equivalent

    score = (
        used * 1e12
        + dist_m * 1e3
        + wait_min
        + frozen_dry * FROZEN_SOFT_PENALTY
    )

    metrics = {
        "pass": True,
        "drop": drop,
        "dispatch_rate": float(dispatch_rate),
        "frozen_dry": frozen_dry,
        "used": used,
        "km": float(km),
        "dist_m": float(dist_m),
        "wait_min": wait_min,
    }
    return float(score), metrics


# -----------------------------
# 1 episode
# -----------------------------
def run_one_episode(params, seed: int):
    init_globals()
    sim = _SIM
    time_matrix = _TIME_MATRIX
    dist_matrix = _DIST_MATRIX

    random.seed(seed)
    np.random.seed(seed)

    # Force fixed order count for fairness
    old_min = cfg.config["simulator"]["ORDER_MIN"]
    old_max = cfg.config["simulator"]["ORDER_MAX"]
    try:
        cfg.config["simulator"]["ORDER_MIN"] = ORDER_N
        cfg.config["simulator"]["ORDER_MAX"] = ORDER_N
        df = sim.generate_orders(save=False)
    finally:
        cfg.config["simulator"]["ORDER_MIN"] = old_min
        cfg.config["simulator"]["ORDER_MAX"] = old_max

    res = solve_pdptw(
        df=df,
        time_matrix=time_matrix,
        dist_matrix=dist_matrix,
        trucks_path=str(cfg.TRUCKS_PATH),
        depot_hub=0,

        horizon=24 * 60,
        time_limit_sec=TIME_LIMIT_SEC,

        allow_drop=ALLOW_DROP,
        drop_penalty=DROP_PENALTY,

        vehicle_fixed_cost=int(params["vehicle_fixed_cost"]),

        frozen_dry_penalty=FROZEN_DRY_PENALTY,

        objective=OBJECTIVE,
        wait_cost_coeff=int(params["wait_cost_coeff"]),

        capacity_scale=int(cfg.config["vehicle"]["CAPACITY_SCALE"]),
    )

    score, metrics = evaluate_with_gate(res, df, dist_matrix, order_n=ORDER_N)
    return score, metrics


def _median(vals):
    vals = [v for v in vals if v is not None]
    return float(statistics.median(vals)) if vals else None


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return float(statistics.mean(vals)) if vals else None


# -----------------------------
# Optuna objective
# -----------------------------
def optuna_objective(trial: optuna.Trial):
    # Tune knobs (policy costs are fixed)
    params = {
        "vehicle_fixed_cost": trial.suggest_int("vehicle_fixed_cost", 150_000, 600_000, log=True),
        "wait_cost_coeff": trial.suggest_int("wait_cost_coeff", 0, 800, step=50),
    }

    scores = []
    metrics_list = []

    for step_i, seed in enumerate(EVAL_SEEDS):
        score, m = run_one_episode(params, seed)
        scores.append(score)
        metrics_list.append(m)

        # report running median for pruning
        trial.report(_median(scores), step=step_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial_value = _median(scores)

    # Helpful aggregates
    pass_flags = [m.get("pass") for m in metrics_list]
    pass_rate = _avg([1.0 if p else 0.0 for p in pass_flags])

    trial.set_user_attr("median_score", trial_value)
    trial.set_user_attr("pass_rate", pass_rate)
    trial.set_user_attr("median_dispatch_rate", _median([m.get("dispatch_rate") for m in metrics_list]))
    trial.set_user_attr("median_drop", _median([m.get("drop") for m in metrics_list]))
    trial.set_user_attr("median_used", _median([m.get("used") for m in metrics_list]))
    trial.set_user_attr("median_km", _median([m.get("km") for m in metrics_list]))
    trial.set_user_attr("median_wait_min", _median([m.get("wait_min") for m in metrics_list]))
    trial.set_user_attr("median_frozen_dry", _median([m.get("frozen_dry") for m in metrics_list]))

    # Log JSONL
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "trial": trial.number,
        "value": trial_value,
        "params": params,
        "user_attrs": dict(trial.user_attrs),
        "order_n": ORDER_N,
        "seeds": EVAL_SEEDS,
        "gate": {
            "min_dispatch_rate": MIN_DISPATCH_RATE,
            "max_drop_ratio": MAX_DROP_RATIO,
            "soft_max_frozen_dry": SOFT_MAX_FROZEN_DRY,
        },
        "policy": {
            "drop_penalty": DROP_PENALTY,
            "frozen_dry_penalty": FROZEN_DRY_PENALTY,
        },
    }
    with open(KPI_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return float(trial_value)  # minimize


if __name__ == "__main__":
    init_globals()

    sampler = optuna.samplers.TPESampler(seed=BASE_SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    storage = "sqlite:///optuna.db"
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name="dispatch_param_tuning_gate",
        load_if_exists=True,
    )

    study.optimize(optuna_objective, n_trials=N_TRIALS)

    print("\n================ BEST ================")
    print("best_value:", study.best_value)
    print("best_params:", study.best_params)
    print("best_attrs:", study.best_trial.user_attrs)
    print("=====================================\n")

    best = study.best_params
    print("Paste into config.py -> config['params']:")
    print("{")
    print(f"  'time_limit_sec': {TIME_LIMIT_SEC},")
    print(f"  'drop_penalty': {DROP_PENALTY},")
    print(f"  'frozen_dry_penalty': {FROZEN_DRY_PENALTY},")
    print(f"  'vehicle_fixed_cost': {best['vehicle_fixed_cost']},")
    print(f"  'wait_cost_coeff': {best['wait_cost_coeff']},")
    print("}")
    print(f"\nSaved Optuna storage: {storage}")
    print(f"Saved KPI log(jsonl): {KPI_LOG_PATH}")

# -----------------------------
# 실행
# -----------------------------
# & "C:\Users\User\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\optuna-dashboard.exe" sqlite:///optuna.db 