import pandas as pd
import random
import os
import pickle
from datetime import datetime, timedelta
import config as cfg


class Simulator:
    def __init__(self):
        self.hub_df = pd.read_excel(cfg.HUB_PATH)
        self.hub_ids = list(self.hub_df.loc[1:, "ID"])

        with open(os.path.join(cfg.DATA_PATH, "dist_matrix.pkl"), "rb") as f:
            self.dist_matrix = pickle.load(f)

        with open(os.path.join(cfg.DATA_PATH, "time_matrix.pkl"), "rb") as f:
            self.time_matrix = pickle.load(f)

    # --------------------------------------------------
    # Pick time generation
    # --------------------------------------------------
    @staticmethod
    def generate_random_pick_time():
        """PU: 09~15시 사이"""
        hour = random.randint(9, 15)
        minute = random.randint(0, 59)
        return datetime.strptime(f"{hour}:{minute}", "%H:%M")

    # --------------------------------------------------
    # Min slack by duration
    # --------------------------------------------------
    @staticmethod
    def min_slack_by_dur(dur_min: float) -> int:
        if dur_min <= 15:
            return 20
        elif dur_min <= 60:
            return 15
        else:
            return 10

    # --------------------------------------------------
    # Order generation
    # --------------------------------------------------
    def generate_orders(self, save: bool = False, filename: str = None):
        
        n_orders = random.randint(
            cfg.config["simulator"]["ORDER_MIN"],
            cfg.config["simulator"]["ORDER_MAX"]
        )
        
        order_list = []
        max_minutes = 7 * 60
        order_date = datetime.today().strftime("%Y-%m-%d")

        # ----------------------------
        # Pricing params
        # ----------------------------
        UNIT_WEIGHT = {        # TYPE별 가중치
            "unit": 1.0,
            "pack": 1.5,
            "box": 4.0,
        }

        TYPE_MULT = {
            "Dry": 1.00,
            "Frozen": 1.25,
            "KKzap": 1.10,
            "Jmart": 1.05,
            "TP_2_PSN": 1.15,
        }

        BASE_FEE = 3000
        PER_KM = 900
        BASE_CAPACITY = 400

        for idx in range(1, n_orders + 1):
            # ----------------------------
            # HUB sampling
            # ----------------------------
            pu_hub_id, do_hub_id = random.sample(self.hub_ids, 2)

            pu_row = self.hub_df[self.hub_df["ID"] == pu_hub_id].iloc[0]
            do_row = self.hub_df[self.hub_df["ID"] == do_hub_id].iloc[0]

            # ----------------------------
            # PU / DO time
            # ----------------------------
            pu_time = self.generate_random_pick_time()

            dist = float(self.dist_matrix[pu_hub_id][do_hub_id])
            dur_base = float(self.time_matrix[pu_hub_id][do_hub_id])

            slack_min = self.min_slack_by_dur(dur_base)
            min_minutes = max(dur_base * 1.2, dur_base + slack_min)

            deadline_dur = random.uniform(min_minutes, max_minutes)
            do_time = pu_time + timedelta(minutes=deadline_dur)

            slack = deadline_dur - dur_base

            # ----------------------------
            # ORDER TIME
            # PU 기준 30~180분 전 주문 발생
            # ----------------------------
            order_time_offset = random.randint(30, 180)
            order_time = pu_time - timedelta(minutes=order_time_offset)

            if order_time < datetime.strptime("00:00", "%H:%M"):
                order_time = datetime.strptime("00:00", "%H:%M")

            # ----------------------------
            # Volume / type
            # ----------------------------
            amount = random.randint(1, cfg.config["simulator"]["AMOUNT_MAX"])
            unit = random.choice(cfg.config["simulator"]["UNIT"])
            delivery_type = random.choice(cfg.config["simulator"]["DELIVERY_TYPE"])

            capacity = amount * UNIT_WEIGHT[unit]

            # ----------------------------
            # Pricing
            # ----------------------------
            raw_cost = (
                BASE_FEE
                + PER_KM * dist
                + BASE_CAPACITY * capacity
            )

            if slack < 30:
                urgency = 1.15
            elif slack < 60:
                urgency = 1.07
            else:
                urgency = 1.00

            price = round(raw_cost * TYPE_MULT[delivery_type] * urgency, -2)

            # ----------------------------
            # Order record
            # ----------------------------
            order = {
                "ORDER_ID": idx,
                "ORDER_DATE": order_date,

                # Online dispatch time
                "ORDER_TIME": order_time.strftime("%H:%M"),
                "ORDER_TIME_IDX": order_time.hour * 60 + order_time.minute,

                # Pickup
                "PU_HUB": int(pu_hub_id),
                "PU_LAT": float(pu_row["lat"]),
                "PU_LNG": float(pu_row["lng"]),
                "PU_TIME": pu_time.strftime("%H:%M"),
                "PU_TIME_IDX": pu_time.hour * 60 + pu_time.minute,

                # Dropoff
                "DO_HUB": int(do_hub_id),
                "DO_LAT": float(do_row["lat"]),
                "DO_LNG": float(do_row["lng"]),
                "DO_TIME": do_time.strftime("%H:%M"),
                "DO_TIME_IDX": do_time.hour * 60 + do_time.minute,

                # Distance / time
                "DIST": round(dist, 3),
                "DUR": round(dur_base, 1),
                "DEADLINE_DUR": round(deadline_dur, 1),
                "SLACK": round(slack, 1),

                # Load
                "AMOUNT": amount,
                "UNIT": unit,
                "DELIVERY_TYPE": delivery_type,
                "CAPACITY": round(capacity, 2),

                # Revenue
                "PRICE": price,
            }

            order_list.append(order)

        df = pd.DataFrame(order_list)

        # ----------------------------
        # Save
        # ----------------------------
        if save:
            if filename is None:
                time_uid = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"simulations_{time_uid}.xlsx"

            os.makedirs(cfg.SIMULATION_PATH, exist_ok=True)
            save_path = os.path.join(cfg.SIMULATION_PATH, filename)
            df.to_excel(save_path, index=False)
            print(f"✅ 시뮬레이션 데이터 생성 완료: {save_path}")
            print("생성된 주문 개수 :", len(df))

        return df
    
    
# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    sim = Simulator()
    n = random.randint(
        cfg.config["simulator"]["ORDER_MIN"],
        cfg.config["simulator"]["ORDER_MAX"]
    )
    df = sim.generate_orders(save=True)