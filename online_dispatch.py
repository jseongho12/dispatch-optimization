# online_dispatch.py
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# 동일 feature builder를 재사용(복붙 or import)
def build_order_features_one(order_row: Dict[str, Any], time_mx, dist_mx) -> pd.DataFrame:
    # order_row must have: PU_HUB, DO_HUB, CAPACITY, PU_TIME_IDX, DO_TIME_IDX, DELIVERY_TYPE
    pu = int(order_row["PU_HUB"])
    do = int(order_row["DO_HUB"])

    cap = float(order_row["CAPACITY"])
    pu_t = int(order_row["PU_TIME_IDX"])
    do_t = int(order_row["DO_TIME_IDX"])
    is_frozen = int(str(order_row.get("DELIVERY_TYPE", "")).strip() == "Frozen")

    direct_time = int(time_mx[pu][do])
    direct_dist = float(dist_mx[pu][do])

    tw_slack = max(0, do_t - pu_t)
    pu_hour = max(0, min(23, pu_t // 60))
    do_hour = max(0, min(23, do_t // 60))

    feat = {
        "PU_HUB": pu,
        "DO_HUB": do,
        "cap": cap,
        "cap_log1p": np.log1p(cap),
        "PU_TIME_IDX": pu_t,
        "DO_TIME_IDX": do_t,
        "tw_slack_min": tw_slack,
        "pu_hour": pu_hour,
        "do_hour": do_hour,
        "direct_time": direct_time,
        "direct_dist": direct_dist,
        "is_frozen_order": is_frozen,
    }
    return pd.DataFrame([feat])


@dataclass
class Stop:
    hub: int
    earliest: int  # earliest service start
    latest: int    # latest service start (or arrival)
    demand: float  # + for PU, - for DO
    order_id: int
    action: str    # "PU" or "DO"


@dataclass
class VehicleState:
    vehicle_id: int
    is_frozen: bool
    capacity: float
    start_hub: int
    start_time: int
    route: List[Stop] = field(default_factory=list)


def simulate_route_cost_and_feasibility(
    veh: VehicleState,
    time_mx,
    dist_mx,
) -> Tuple[bool, float, int, float]:
    """
    간단 시뮬레이션:
      - 시간창: arrival_time must be <= latest, service start >= earliest
      - 용량: 누적 load in [0, capacity]
    Returns: feasible, total_dist, end_time, total_time
    """
    t = int(veh.start_time)
    hub = int(veh.start_hub)
    load = 0.0
    total_dist = 0.0
    total_time = 0.0

    for st in veh.route:
        travel_t = int(time_mx[hub][st.hub])
        travel_d = float(dist_mx[hub][st.hub])
        t += travel_t
        total_time += travel_t
        total_dist += travel_d

        # wait until earliest
        if t < st.earliest:
            t = st.earliest

        # time window violation
        if t > st.latest:
            return False, 1e18, 10**9, 1e18

        # apply demand
        load += float(st.demand)
        if load < -1e-6 or load - veh.capacity > 1e-6:
            return False, 1e18, 10**9, 1e18

        hub = st.hub

    return True, total_dist, t, total_time


def best_insertion_for_order(
    veh: VehicleState,
    order_id: int,
    pu_hub: int, do_hub: int,
    pu_t: int, do_t: int,
    demand: float,
    time_mx, dist_mx,
) -> Tuple[bool, float, List[Stop]]:
    """
    (PU, DO) 두 스톱을 route 리스트에 끼워넣는 모든 위치를 탐색(느리지만 차량별 route가 짧으면 OK)
    Returns: feasible, best_extra_dist, best_route
    """
    # 원래 비용
    ok0, d0, _, _ = simulate_route_cost_and_feasibility(veh, time_mx, dist_mx)
    if not ok0:
        return False, 1e18, veh.route

    pu = Stop(hub=pu_hub, earliest=pu_t, latest=24*60, demand=+demand, order_id=order_id, action="PU")
    do = Stop(hub=do_hub, earliest=0, latest=do_t, demand=-demand, order_id=order_id, action="DO")

    best = (False, 1e18, None)

    n = len(veh.route)
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            cand = veh.route.copy()
            cand.insert(i, pu)
            cand.insert(j, do)

            tmp = VehicleState(**{**veh.__dict__, "route": cand})
            ok, d1, _, _ = simulate_route_cost_and_feasibility(tmp, time_mx, dist_mx)
            if not ok:
                continue
            extra = d1 - d0
            if extra < best[1]:
                best = (True, extra, cand)

    if best[2] is None:
        return False, 1e18, veh.route
    return best[0], best[1], best[2]


class OnlineDispatcher:
    def __init__(self, clf_path: str, reg_path: Optional[str] = None):
        self.clf = joblib.load(clf_path)
        self.reg = joblib.load(reg_path) if reg_path else None

    def decide_and_insert(
        self,
        order_row: Dict[str, Any],
        vehicles: List[VehicleState],
        time_mx,
        dist_mx,
        accept_threshold: float = 0.55,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """
        1) 분류: accept/reject
        2) (옵션) 회귀: 차량 후보 랭킹 (Δobj 낮을수록 좋다 가정)
        3) 삽입 휴리스틱으로 실제 반영
        """
        X = build_order_features_one(order_row, time_mx, dist_mx)
        p_accept = float(self.clf.predict_proba(X)[:, 1][0])

        if p_accept < accept_threshold:
            return {"accepted": False, "reason": "clf_reject", "p_accept": p_accept}

        order_id = int(order_row.get("ORDER_ID", -1))
        pu_hub = int(order_row["PU_HUB"])
        do_hub = int(order_row["DO_HUB"])
        pu_t = int(order_row["PU_TIME_IDX"])
        do_t = int(order_row["DO_TIME_IDX"])
        demand = float(order_row["CAPACITY"])
        is_frozen_order = (str(order_row.get("DELIVERY_TYPE", "")).strip() == "Frozen")

        # 차량 후보 필터: Frozen 주문은 Frozen 차량만
        cand_vehicles = [v for v in vehicles if (v.is_frozen or not is_frozen_order)]
        if not cand_vehicles:
            return {"accepted": False, "reason": "no_vehicle_type", "p_accept": p_accept}

        # 회귀가 있으면 후보 랭킹, 없으면 전부 탐색
        if self.reg is not None:
            # 차량 특징을 추가하려면 feature를 확장해야 하지만 MVP라 order-only 회귀 점수로만 후보 수를 줄임
            score = float(self.reg.predict(X)[0])  # 낮을수록 좋다고 가정(Δobj)
            # score는 차량별로 다르지 않아서 여기서는 “탐색 top_k 제한” 용도만 수행
            cand_vehicles = cand_vehicles[: min(top_k, len(cand_vehicles))]

        best = None
        for v in cand_vehicles:
            ok, extra_dist, new_route = best_insertion_for_order(
                v, order_id, pu_hub, do_hub, pu_t, do_t, demand, time_mx, dist_mx
            )
            if not ok:
                continue
            if best is None or extra_dist < best["extra_dist"]:
                best = {"vehicle_id": v.vehicle_id, "extra_dist": extra_dist, "new_route": new_route}

        if best is None:
            return {"accepted": False, "reason": "no_feasible_insertion", "p_accept": p_accept}

        # 반영
        vid = best["vehicle_id"]
        for v in vehicles:
            if v.vehicle_id == vid:
                v.route = best["new_route"]
                break

        return {
            "accepted": True,
            "p_accept": p_accept,
            "vehicle_id": vid,
            "extra_dist": float(best["extra_dist"]),
            "route_len": int(len(best["new_route"])),
        }

if __name__ == "__main__":
    import os
    import pandas as pd
    from simulator import Simulator
    import config as cfg

    # 1) 모델 로드
    disp = OnlineDispatcher(
        clf_path="ml_models/clf_assigned.joblib",
        reg_path=None,  # 회귀 아직 없으니까 None
    )

    # 2) 시뮬레이터로 주문 생성
    sim = Simulator()
    df = sim.generate_orders(save=False)
    time_mx = sim.time_matrix
    dist_mx = sim.dist_matrix

    # 3) 차량 상태 만들기 (trucks.xlsx에서)
    trucks_path = os.path.join(cfg.DATA_PATH, "trucks.xlsx")
    trucks_df = pd.read_excel(trucks_path)

    vehicles = []
    for vid, r in trucks_df.iterrows():
        # Frozen 트럭 판별은 solver 내부 로직과 맞추는 게 이상적이지만 MVP로는 문자열 포함으로 처리
        truck_type = str(r.get("TRUCK_TYPE", "")).lower()
        is_frozen = ("frozen" in truck_type) or ("냉동" in truck_type)

        vehicles.append(
            VehicleState(
                vehicle_id=int(vid),
                is_frozen=bool(is_frozen),
                capacity=float(r["TRUCK_VOLUME"]),
                start_hub=0,
                start_time=0,
            )
        )

    # 4) 주문이 발생 시간 순서로 들어온다고 가정하고 10개만 흘려보내기
    df = df.sort_values("ORDER_TIME_IDX").reset_index(drop=True)

    accepted = 0
    for i in range(min(10, len(df))):
        order = df.loc[i].to_dict()

        res = disp.decide_and_insert(
            order_row=order,
            vehicles=vehicles,
            time_mx=time_mx,
            dist_mx=dist_mx,
            accept_threshold=0.30,  # 처음엔 낮게
            top_k=8,
        )

        print(f"[ORDER {int(order['ORDER_ID'])}] {res}")
        accepted += int(res.get("accepted", False))

    print(f"\nDONE. accepted={accepted}/{min(10, len(df))}")
