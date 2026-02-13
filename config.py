from pathlib import Path
from datetime import datetime

ROOT_DIR  = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "datas"
SIMULATION_PATH = ROOT_DIR / "simulations"
HUB_PATH  = DATA_PATH / "hubs.xlsx"
TRUCKS_PATH = DATA_PATH / "trucks.xlsx"

OSRM_URL = "http://router.project-osrm.org"

DATE = datetime.now().strftime("%Y-%m-%d")

config = {
    'simulator': {
        "DATE": DATE.replace('-', ''),
        "AMOUNT_MAX": 5,
        "UNIT": ["pack", "unit", "box"],
        "DELIVERY_TYPE": ["Dry", "Frozen", "KKzap", "Jmart", "TP_2_PSN"],
        "ORDER_MIN": 15,
        "ORDER_MAX": 40,
    },
    
    'vehicle': {
        "NUM": 1,
        "DEFAULT_COUNT": 3,
        "CAPACITY_SCALE": 100,
    },
    
    # 'params': {
    #     "time_limit_sec": 10,          # solver 최적해 도출 시간
    #     "drop_penalty": 3_000,         # drop 1건 = 300km 추가와 동급
    #     "vehicle_fixed_cost": 0,       # 차량 1대 = 120km 추가와 동급
    #     "wait_cost_coeff": 0,          # 대기 1분 = 0m 추가와 동급 (작게 시작)
    #     "frozen_dry_penalty": 500      # 냉동차가 dry 처리 1건 = 50km 추가와 동급
    # },
    
    
    
    # optuna
    
    'params': {
        'time_limit_sec': 10,
        'drop_penalty': 10_000_000,          #1) 배차 최대 (수익 최대)
        'frozen_dry_penalty': 10_000_000,    #2) 냉동 차량 페널티
        
        'vehicle_fixed_cost': 556_058,       #3) 차량 최소
        'wait_cost_coeff': 0,                #4) 대기 최소
    }
    
    
    # 'params': {
    #     'time_limit_sec': 10,
    #     'drop_penalty': 10_000_000,          #1) 배차 최대 (수익 최대)
    #     'frozen_dry_penalty': 10_000_000,    #2) 냉동 차량 페널티
        
    #     'vehicle_fixed_cost': 240_000,       #3) 차량 최소
    #     'wait_cost_coeff': 200,              #4) 대기 최소
    # }
    

}
