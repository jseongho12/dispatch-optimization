import os
import glob
import pickle
import pandas as pd
import config as cfg
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from simulator import Simulator


# ---------------------------
# Utils
# ---------------------------
def _min_to_hhmm(m: int) -> str:
    m = int(m)
    h = (m // 60) % 24
    mm = m % 60
    return f"{h:02d}:{mm:02d}"


def _is_frozen_truck(row: pd.Series) -> bool:
    # trucks.xlsx의 TRUCK_TYPE에 '냉동' 포함이면 냉동차량
    return "냉동" in str(row.get("TRUCK_TYPE", ""))


def _is_frozen_order(row: pd.Series) -> bool:
    # 주문의 DELIVERY_TYPE == "Frozen" 이면 냉동 주문
    return str(row.get("DELIVERY_TYPE", "")).strip() == "Frozen"


def load_trucks_from_excel(
    trucks_path: str,
) -> pd.DataFrame:
    """
    trucks.xlsx를 읽어서 fleet DataFrame 반환.
    - 기본: 엑셀의 한 행 = 차량 1대
    - COUNT만큼 행을 복제해서 '차량 리스트'로 확장
    """
    trucks = pd.read_excel(trucks_path).copy()

    trucks = trucks.loc[trucks.index.repeat(trucks["COUNT"])].reset_index(drop=True)
    trucks = trucks.drop(columns=["COUNT"])

    return trucks


# ---------------------------
# OR-Tools PDPTW
# ---------------------------
def solve_pdptw(
    df: pd.DataFrame,
    time_matrix,
    dist_matrix,
    trucks_path: str,
    depot_hub: int = 0,

    # 시간 관련
    horizon: int = 24 * 60,
    time_limit_sec: int = 10,

    # (고정) 배차 최대
    allow_drop: bool = True,
    drop_penalty: int = 10_000_000,

    # (고정) 냉동차가 Dry 주문을 수행하면 페널티(arc cost에 추가)
    frozen_dry_penalty: int = 10_000_000,
    
    # (선택) 1) 차량 최소
    vehicle_fixed_cost: int = 556_058,

    # (선택) 2) 거리 최소 (시간은 영향을 많이 받음)
    objective: str = "dist",

    # (선택) 3) 대기 최소화(0이면 OFF)
    wait_cost_coeff: int = 0,

    # 용량 소수 보존
    capacity_scale: int = 100,
):
    """
    우선순위 설계:
    (1) 배차 최대: drop_penalty 크게 (allow_drop=True)
    (2) 차량 최소: vehicle_fixed_cost > 0
    (3) 거리/시간: objective로 선택
    (4) 대기 최소: wait_cost_coeff로 선택 (0=OFF)
    """

    # -----------------------
    # Load fleet
    # -----------------------
    trucks_df = load_trucks_from_excel(trucks_path)
    num_vehicles = len(trucks_df)
    
    # 차량별 용량: TRUCK_VOLUME 기반 (소수 보존 위해 scale)
    if "TRUCK_VOLUME" not in trucks_df.columns:
        raise KeyError("trucks.xlsx must contain TRUCK_VOLUME column")

    vehicle_capacities = (
        trucks_df["TRUCK_VOLUME"].astype(float) * float(capacity_scale)
    ).round().astype(int).tolist()

    frozen_vehicle_indices = [i for i, r in trucks_df.iterrows() if _is_frozen_truck(r)]
    all_vehicle_indices = list(range(num_vehicles))

    # -----------------------
    # Build nodes
    # -----------------------
    df = df.reset_index(drop=True)
    n_orders = len(df)
    num_nodes = 1 + 2 * n_orders  # depot + PU/DO pairs

    node_hub = [depot_hub] * num_nodes
    demands = [0] * num_nodes
    tw_start = [0] * num_nodes
    tw_end = [horizon] * num_nodes

    # depot TW
    tw_start[0], tw_end[0] = 0, horizon

    is_frozen = [False] * n_orders

    for i, row in df.iterrows():
        pu_node = 1 + 2 * i
        do_node = 1 + 2 * i + 1

        pu_hub = int(row["PU_HUB"])
        do_hub = int(row["DO_HUB"])
        node_hub[pu_node] = pu_hub
        node_hub[do_node] = do_hub

        # ✅ 주문 용량: CAPACITY * scale (소수 보존)
        cap_scaled = int(round(float(row["CAPACITY"]) * float(capacity_scale)))
        demands[pu_node] = +cap_scaled
        demands[do_node] = -cap_scaled

        pu_t = int(row["PU_TIME_IDX"])
        do_t = int(row["DO_TIME_IDX"])

        # (기존 로직 유지)
        tw_start[pu_node] = max(0, pu_t)
        tw_end[pu_node] = horizon

        tw_start[do_node] = 0
        tw_end[do_node] = min(horizon, do_t)

        is_frozen[i] = _is_frozen_order(row)

    # -----------------------
    # Routing model
    # -----------------------
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # transit time callback
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        from_h = node_hub[from_node]
        to_h = node_hub[to_node]
        return int(time_matrix[from_h][to_h])

    time_cb_idx = routing.RegisterTransitCallback(time_callback)

    # (선택) objective + (선택) 냉동차->Dry 페널티(차량별 arc cost)
    # - OR-Tools는 차량별 비용을 주려면 evaluator를 vehicle마다 따로 설정해야 함
    # - 페널티는 "Dry 주문의 PU"에만 1회 부과(중복 페널티 방지)
    is_frozen_vehicle = [i in set(frozen_vehicle_indices) for i in range(num_vehicles)]

    if objective == "dist":
        def base_cost(from_node: int, to_node: int) -> int:
            from_h = node_hub[from_node]
            to_h = node_hub[to_node]
            return int(dist_matrix[from_h][to_h] * 1000)  # km -> m(정수)
    else:
        def base_cost(from_node: int, to_node: int) -> int:
            from_h = node_hub[from_node]
            to_h = node_hub[to_node]
            return int(time_matrix[from_h][to_h])

    # node -> (order_i, is_pickup) map (패널티 계산에 사용)
    node_to_order = {}
    for i in range(n_orders):
        pu_node = 1 + 2 * i
        do_node = 1 + 2 * i + 1
        node_to_order[pu_node] = (i, True)
        node_to_order[do_node] = (i, False)

    for v in range(num_vehicles):
        def make_vehicle_cb(vv: int):
            def vehicle_cost_cb(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)

                cost = base_cost(from_node, to_node)

                # ✅ 냉동차가 Dry 주문을 수행(= 해당 주문의 PU 방문)하면 페널티 추가
                if frozen_dry_penalty and frozen_dry_penalty > 0 and is_frozen_vehicle[vv]:
                    info = node_to_order.get(to_node)
                    if info is not None:
                        oi, is_pu = info
                        if is_pu and (not is_frozen[oi]):  # Frozen 주문이 아니면 Dry 취급
                            cost += int(frozen_dry_penalty)

                return int(cost)
            return vehicle_cost_cb

        cb_idx = routing.RegisterTransitCallback(make_vehicle_cb(v))
        routing.SetArcCostEvaluatorOfVehicle(cb_idx, v)


    # ✅ (고정) 차량 최소: 차량 사용 고정비용
    if vehicle_fixed_cost and vehicle_fixed_cost > 0:
        for v in range(num_vehicles):
            routing.SetFixedCostOfVehicle(int(vehicle_fixed_cost), v)

    # Time dimension (waiting slack 충분히 크게)
    routing.AddDimension(
        time_cb_idx,
        horizon,   # waiting slack max
        horizon,   # max time
        False,     # start cumul 0 강제 X
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    for node in range(num_nodes):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(tw_start[node], tw_end[node])

    # ✅ (선택) 대기 최소화: slack에 비용 부여
    if wait_cost_coeff > 0:
        time_dim.SetSpanCostCoefficientForAllVehicles(int(wait_cost_coeff))

    # Capacity dimension
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return int(demands[node])

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)

    # ✅ 차량별 용량 적용
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,  # capacity slack
        vehicle_capacities,
        True,  # start cumul 0
        "Capacity"
    )
    cap_dim = routing.GetDimensionOrDie("Capacity")

    # Pickup & Dropoff constraints + Frozen 차량 제한
    for i in range(n_orders):
        pu_node = 1 + 2 * i
        do_node = 1 + 2 * i + 1

        pu_idx = manager.NodeToIndex(pu_node)
        do_idx = manager.NodeToIndex(do_node)

        routing.AddPickupAndDelivery(pu_idx, do_idx)
        routing.solver().Add(routing.VehicleVar(pu_idx) == routing.VehicleVar(do_idx))
        routing.solver().Add(time_dim.CumulVar(pu_idx) <= time_dim.CumulVar(do_idx))

        # ✅ Frozen 주문이면 냉동 트럭만 허용
        if is_frozen[i]:
            if not frozen_vehicle_indices:
                # 냉동 트럭이 없으면: 드랍 허용이면 드랍으로만 처리, 아니면 infeasible
                if not allow_drop:
                    return None
            else:
                routing.SetAllowedVehiclesForIndex(frozen_vehicle_indices, pu_idx)
                routing.SetAllowedVehiclesForIndex(frozen_vehicle_indices, do_idx)
        else:
            routing.SetAllowedVehiclesForIndex(all_vehicle_indices, pu_idx)
            routing.SetAllowedVehiclesForIndex(all_vehicle_indices, do_idx)

        # (고정) 배차 최대: 드랍 페널티 크게
        if allow_drop:
            # ✅ 주문 1건 드랍 = 페널티 1회만 (PU에만 부과)
            routing.AddDisjunction([pu_idx], drop_penalty)
            # DO는 '드랍 가능'하게 만들되, 페널티는 0으로 두고 PU와 활성화 상태를 묶는다
            routing.AddDisjunction([do_idx], 0)
            routing.solver().Add(routing.ActiveVar(pu_idx) == routing.ActiveVar(do_idx))
# Solve
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit_sec))

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None

    # Parse solution
    def node_to_order_action(node: int):
        if node == 0:
            return None, None, "DEPOT"
        order_i = (node - 1) // 2
        action = "PU" if (node % 2 == 1) else "DO"
        order_id = int(df.loc[order_i, "ORDER_ID"]) if "ORDER_ID" in df.columns else (order_i + 1)
        return order_i, order_id, action

    routes = []
    assigned_orders = set()

    for v in range(num_vehicles):
        idx = routing.Start(v)
        veh_steps = []

        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            active = sol.Value(routing.ActiveVar(idx)) if allow_drop else 1

            arr = sol.Value(time_dim.CumulVar(idx))
            load = sol.Value(cap_dim.CumulVar(idx))

            order_i, order_id, action = node_to_order_action(node)

            tw_s, tw_e = tw_start[node], tw_end[node]
            tw_ok = (tw_s <= arr <= tw_e)

            veh_steps.append({
                "vehicle": v,
                "node": node,
                "hub": int(node_hub[node]),
                "active": int(active),
                "order_i": order_i,
                "order_id": order_id,
                "action": action,
                "arr_min": int(arr),
                "arr_hhmm": _min_to_hhmm(arr),
                "tw_start_min": int(tw_s),
                "tw_start_hhmm": _min_to_hhmm(tw_s),
                "tw_end_min": int(tw_e),
                "tw_end_hhmm": _min_to_hhmm(tw_e),
                "tw_ok": bool(tw_ok),

                # ✅ load는 scale된 정수와, 사람이 보기 좋은 실수 둘 다
                "load_scaled": int(load),
                "load": float(load) / float(capacity_scale),
            })

            if active == 1 and order_i is not None:
                assigned_orders.add(order_i)

            idx = sol.Value(routing.NextVar(idx))

        # end
        node = manager.IndexToNode(idx)
        arr = sol.Value(time_dim.CumulVar(idx))
        load = sol.Value(cap_dim.CumulVar(idx))

        veh_steps.append({
            "vehicle": v,
            "node": node,
            "hub": int(depot_hub),
            "active": 1,
            "order_i": None,
            "order_id": None,
            "action": "END",
            "arr_min": int(arr),
            "arr_hhmm": _min_to_hhmm(arr),
            "tw_start_min": 0,
            "tw_start_hhmm": _min_to_hhmm(0),
            "tw_end_min": horizon,
            "tw_end_hhmm": _min_to_hhmm(horizon),
            "tw_ok": True,
            "load_scaled": int(load),
            "load": float(load) / float(capacity_scale),
        })

        routes.append(veh_steps)

    unassigned = [i for i in range(n_orders) if i not in assigned_orders]

    # 사용한 차량 수(업무 있는 차량)
    used_vehicles = 0
    for route in routes:
        if any(step["action"] in ("PU", "DO") and step["active"] == 1 for step in route):
            used_vehicles += 1

    return {
        "routes": routes,
        "unassigned_orders": unassigned,
        "objective": int(sol.ObjectiveValue()),
        "num_vehicles": int(num_vehicles),
        "used_vehicles": int(used_vehicles),
        "capacity_scale": int(capacity_scale),
        "frozen_vehicle_indices": frozen_vehicle_indices,
    }


# ---------------------------
# Log print (기존 형식 유지 + scale 반영)
# ---------------------------
def log_print(
    res,
    df_all,
    time_matrix,
    dist_matrix,
    horizon: int = 24 * 60,
    unit_korean: bool = True
):
    if res is None:
        print("❌ No solution")
        return

    scale = int(res.get("capacity_scale", 1))

    df_all = df_all.copy()
    if "ORDER_ID" not in df_all.columns:
        df_all["ORDER_ID"] = range(1, len(df_all) + 1)
    order_map = df_all.set_index("ORDER_ID")

    cap_map = order_map["CAPACITY"].to_dict() if "CAPACITY" in order_map.columns else {}
    price_map = order_map["PRICE"].to_dict() if "PRICE" in order_map.columns else {}

    n_total_orders = len(df_all)

    time_unit = "분" if unit_korean else "min"
    dist_unit = "km"

    print(f"✅ SOLVED | objective={res.get('objective', 'NA')} | used_vehicles={res.get('used_vehicles','NA')}/{res.get('num_vehicles','NA')}")
    print("unassigned:", res.get("unassigned_orders", []))

    total_done_orders = 0
    total_travel_min = 0
    total_wait_min = 0
    total_km = 0.0
    total_profit = 0

    paid_orders = set()
    vehicle_summaries = []

    for v, route in enumerate(res["routes"]):
        has_job = any(step["action"] in ("PU", "DO") and step.get("active", 1) == 1 for step in route)
        if not has_job:
            continue

        is_frozen_v = v in set(res.get("frozen_vehicle_indices", []))
        tag = "FROZEN" if is_frozen_v else "DRY"
        print(f"--- Vehicle {v} [{tag}] ---")

        header = (
            f"{'STEP':<12} | {'TYPE':<8} | {'hub':>3} | {'ETA':>5} | {'TW':<13} | "
            f"{'load':>12} | {'wait':>6} | {'time':>7} | {'dist':>8} | {'KRW':>9}"
        )

        print(header)
        print("-" * len(header))

        prev_start = None
        prev_hub = None
        first_pu_done = False

        load_sim = 0.0

        veh_done_orders = 0
        veh_travel_min = 0
        veh_wait_min = 0
        veh_km = 0.0
        veh_profit = 0

        for step in route:
            action = step["action"]
            hub = int(step["hub"])
            start = int(step["arr_min"])

            if prev_start is None:
                seg_travel = 0
                seg_km = 0.0
                eta = start
            else:
                seg_travel = int(time_matrix[prev_hub][hub])
                seg_km = float(dist_matrix[prev_hub][hub])
                eta = prev_start + seg_travel

            oid = None
            if action in ("DEPOT", "END"):
                dtype = "-"
                step_label = action
                tw_start = 0
                tw_end = horizon
                tw_str = "-------------"
                demand = 0.0
            else:
                oid = int(step["order_id"])
                row = order_map.loc[oid]
                dtype = str(row.get("DELIVERY_TYPE", "NA"))
                tw_start = int(row["PU_TIME_IDX"])
                tw_end = int(row["DO_TIME_IDX"])
                tw_str = f"{_min_to_hhmm(tw_start)}~{_min_to_hhmm(tw_end)}"
                step_label = f"{action}({oid:>3})"

                cap = float(cap_map.get(oid, 0.0))
                demand = cap if action == "PU" else (-cap if action == "DO" else 0.0)

            waiting = 0 if action in ("DEPOT", "END") else max(0, tw_start - eta)

            eta_disp = eta
            if (not first_pu_done) and action == "PU":
                eta_disp = tw_start
                waiting = 0
                first_pu_done = True

            load_before = load_sim
            load_after = load_sim + demand
            load_sim = load_after
            load_str = f"{load_before:>5.2f}->{load_after:<5.2f}"

            step_price = 0
            if action == "DO" and oid is not None and oid not in paid_orders:
                step_price = int(round(float(price_map.get(oid, 0))))
                paid_orders.add(oid)

                veh_profit += step_price
                total_profit += step_price

                veh_done_orders += 1
                total_done_orders += 1

            veh_travel_min += seg_travel
            veh_wait_min += waiting
            veh_km += seg_km

            total_travel_min += seg_travel
            total_wait_min += waiting
            total_km += seg_km

            wait_str = f"{waiting:>3}{time_unit}"
            time_str = f"{seg_travel:>3}{time_unit}"
            dist_str = f"{seg_km:>6.1f}{dist_unit}"
            price_str = f"{step_price:>9}" if step_price else f"{'':>9}"

            print(
                f"{step_label:<12} | {dtype:<8} | {hub:>3} | {_min_to_hhmm(eta_disp):>5} | {tw_str:<13} | "
                f"{load_str:>12} | {wait_str:>6} | {time_str:>7} | {dist_str:>8} | {price_str}"
            )


            prev_start = start
            prev_hub = hub

        veh_total = veh_travel_min + veh_wait_min
        vehicle_summaries.append({
            "vehicle": v,
            "done_orders": veh_done_orders,
            "travel_dist_km": veh_km,
            "wait_min": veh_wait_min,
            "total_min": veh_total,
            "profit": veh_profit,
        })

    if not vehicle_summaries:
        print("\n(배차된 차량이 없습니다.)")
        return

    total_time_with_wait = total_travel_min + total_wait_min

    print("\n================ SUMMARY ================")
    print(f"{'V':>2} | {'DONE':>4} | {'TravelDist':>10} | {'WaitTime':>8} | {'TotalTime':>9} | {'Profit':>10}")
    print("-" * 62)

    for s in vehicle_summaries:
        print(
            f"{s['vehicle']:>2} | {s['done_orders']:>4} | {s['travel_dist_km']:>9.1f}km | "
            f"{s['wait_min']:>7}{time_unit} | {s['total_min']:>8}{time_unit} | {s['profit']:>10}"
        )

    print("-" * 62)
    print(
        f"{'ALL':>2} | {total_done_orders:>4} | {total_km:>9.1f}km | "
        f"{total_wait_min:>7}{time_unit} | {total_time_with_wait:>8}{time_unit} | {total_profit:>10}"
    )
    print("========================================")

    dispatch_rate = (total_done_orders / n_total_orders) if n_total_orders else 0.0
    wait_rate = (total_wait_min / total_time_with_wait) if total_time_with_wait else 0.0
    profit_per_km = (total_profit / total_km) if total_km > 0 else 0.0
    total_hours = total_time_with_wait / 60.0
    profit_per_hour = (total_profit / total_hours) if total_hours > 0 else 0.0

    print("\n================ OVERALL KPI ================")
    print(f"{'배차율':<14}: {dispatch_rate*100:>6.1f}%   ({total_done_orders}/{n_total_orders})")
    print(f"{'총 이동거리':<14}: {total_km:>10.1f} km")
    print(f"{'총 이동시간':<14}: {total_time_with_wait:>10d} min")
    print(f"{'순수 이동시간':<14}: {total_travel_min:>10d} min")
    print(f"{'총 대기시간':<14}: {total_wait_min:>10d} min")
    print(f"{'대기율':<14}: {wait_rate*100:>6.1f}%   (대기/총시간)")
    print(f"{'총 수익':<14}: {total_profit:>10,} KRW")
    print(f"{'거리당 수익':<14}: {profit_per_km:>10.1f} KRW/km")
    print(f"{'시간당 수익':<14}: {profit_per_hour:>10.1f} KRW/h")
    print("=============================================\n")


# ---------------------------
# Main (latest excel + load matrices)
# ---------------------------
if __name__ == "__main__":
    # 1) matrices
    with open(os.path.join(cfg.DATA_PATH, "time_matrix.pkl"), "rb") as f:
        time_matrix = pickle.load(f)
    with open(os.path.join(cfg.DATA_PATH, "dist_matrix.pkl"), "rb") as f:
        dist_matrix = pickle.load(f)

    # 2) latest simulation excel
    files = glob.glob(os.path.join(cfg.SIMULATION_PATH, "*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx found in: {cfg.SIMULATION_PATH}")
    latest_file = max(files, key=os.path.getctime)
    print("📄 using file:", latest_file)

    df = pd.read_excel(latest_file)

    # 3) solve
    # ✅ trucks.xlsx는 cfg 쪽에서 관리하거나, 여기서 직접 경로 지정
    trucks_path = os.path.join(cfg.TRUCKS_PATH)

    res = solve_pdptw(
        df=df,
        time_matrix=time_matrix,
        dist_matrix=dist_matrix,
        trucks_path=trucks_path,
        depot_hub=0,

        # (고정) 1) 배차 최대 (수익 최대)
        allow_drop=True,
        drop_penalty=cfg.config["params"]["drop_penalty"],
        
        # (고정) 2) 냉동 차량 페널티
        frozen_dry_penalty=cfg.config["params"]["frozen_dry_penalty"],

        # (선택) 3) 차량 최소
        vehicle_fixed_cost=cfg.config["params"]["vehicle_fixed_cost"],

        # (선택) 4) time/dist
        objective="dist",

        # (선택) 5) 대기 최소 (0=OFF, 예: 50=ON)
        wait_cost_coeff=cfg.config["params"]["wait_cost_coeff"],

        time_limit_sec=cfg.config["params"]["time_limit_sec"],
        capacity_scale=100,

    )

    # 4) print
    log_print(res, df, time_matrix, dist_matrix)
