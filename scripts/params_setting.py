import pandas as pd 
import numpy as np
import pickle
import os
import glob


root = 'C:\\Users\\User\\0. 노션 포트폴리오\\개인 배차최적화'
root = root.replace('\\','/')

with open(root + '/datas/dist_matrix.pkl', 'rb') as f:
    dist = pickle.load(f)

def _percentiles(x, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {p: np.nan for p in ps}
    return {p: float(np.percentile(x, p)) for p in ps}

def analyze_dist_matrix(dist, depot=0):
    """
    dist_matrix: (H x H) in km
    - depot->others 분포
    - 전체 hub->hub(대각/0 제외) 분포
    """
    dist = np.asarray(dist, dtype=float)

    # depot -> others (0 제외)
    d0 = dist[depot].copy()
    d0 = d0[(d0 > 0) & np.isfinite(d0)]

    # 전체 (대각/0 제외)
    allv = dist.reshape(-1)
    allv = allv[(allv > 0) & np.isfinite(allv)]

    stats = {
        "depot_min_km": float(np.min(d0)) if d0.size else np.nan,
        "depot_pctl_km": _percentiles(d0),
        "all_pctl_km": _percentiles(allv),
        "n_depot_edges": int(d0.size),
        "n_all_edges": int(allv.size),
    }
    return stats

def suggest_vehicle_fixed_cost_km(stats, base="all_p75", km_range=(50, 800)):
    """
    OR-Tools에서 dist objective는 km*1000 => '미터 단위 정수 비용'으로 들어가니까,
    vehicle_fixed_cost도 (K km) * 1000 으로 맞추면 됨.
    - km_range: '차량 1대 = K km' 가정의 탐색 범위
    """
    # 대표값 참고용(선택)
    depot_p75 = stats["depot_pctl_km"].get(75, np.nan)
    all_p75 = stats["all_pctl_km"].get(75, np.nan)

    print("=== DIST STATS (km) ===")
    print(f"depot->others  min: {stats['depot_min_km']:.4f} km")
    print(f"depot->others  p50: {stats['depot_pctl_km'][50]:.4f} km | p75: {depot_p75:.4f} km | p90: {stats['depot_pctl_km'][90]:.4f} km")
    print(f"all hub->hub   p50: {stats['all_pctl_km'][50]:.4f} km | p75: {all_p75:.4f} km | p90: {stats['all_pctl_km'][90]:.4f} km")
    print()

    lo_km, hi_km = km_range
    lo_cost = int(lo_km * 1000)   # km -> meter-cost
    hi_cost = int(hi_km * 1000)

    print("=== RECOMMENDED SEARCH SPACE ===")
    print(f"[vehicle_fixed_cost]  {lo_cost:,} ~ {hi_cost:,}  (차량 1대 = {lo_km}~{hi_km} km 가정)")
    print(f"[wait_cost_coeff]     0 ~ 2000  (너무 키우면 차량 늘어나는 쪽으로 튈 수 있음)")
    print("(drop_penalty / frozen_dry_penalty는 정책값으로 고정 추천)")
    return lo_cost, hi_cost

if __name__ == "__main__":

    stats = analyze_dist_matrix(dist=dist, depot=0)
    suggest_vehicle_fixed_cost_km(stats, km_range=(50, 800))
