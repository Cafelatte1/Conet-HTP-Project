GLOBAL_SEED = 42

import os
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

import numpy as np # linear algebra
from numpy import random as np_rnd
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import random as rnd
import pickle
import gc
import time
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')

from helper_functions import *

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # python random
    rnd.seed(seed)
    # numpy random
    np_rnd.seed(seed)
    # RAPIDS random
    try:
        cupy.random.seed(seed)
    except:
        pass
    # tf random
    try:
        tf_rnd.set_seed(seed)
    except:
        pass
    # pytorch random
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass

class CFG:
    debug = False

label_mapper = {
    "house": {
        'door_yn':{'n':0, 'y':1}, # 문 유무
        'loc':{'left':0, 'center':1, 'right':2}, # 위치
        'roof_yn':{'y':1, 'n':0}, # 지붕 유무
        'window_cnt':{'absence':0, '1 or 2':1, 'more than 3':2}, # 창문 갯수
        'size':{'small':0, 'middle':1, 'big':2}, # 크기
    },
    "tree": {
        "branch_yn": {"n": 0, "y": 1}, # 가지 유무
        "root_yn": {"n": 0, "y": 1}, # 뿌리 유무
        "crown_yn": {"n": 0, "y": 1}, # 수관 유무
        "fruit_yn": {"n": 0, "y": 1}, # 열매 유무
        "gnarl_yn": {"n": 0, "y": 1}, # 옹이나상처 유무
        "loc": {"left": 0, "center": 1, "right": 2}, # 위치
        "size": {"small": 0, "middle": 1, "big": 2}, # 크기
    },
    "person": {
        "eye_yn": {"n": 0, "y": 1}, # 눈 유무
        "leg_yn": {"n": 0, "y": 1}, # 다리 유무
        "loc": {"left": 0, "center": 1, "right": 2}, # 위치
        "mouth_yn": {"n": 0, "y": 1}, # 입 유무
        "size": {"small": 0, "middle": 1, "big": 2}, # 크기
        "arm_yn": {"n": 0, "y": 1}, # 팔 유무
    }
}

def main():
    df_output = {
        "detectCls": None,
        "singleshot": None,
        "detectCls_resnet": None,
    }
    df_output["detectCls"] = {k: pd.read_csv(f"./detectCls_{k}_output.csv") for k in label_mapper.keys()}
    df_output["singleshot"] = {k: pd.read_csv(f"./singleshot_{k}_output.csv") for k in label_mapper.keys()}
    df_output["detectCls_resnet"] = {k: pd.read_csv(f"./detectCls_resnet_{k}_output.csv") for k in label_mapper.keys()}

    ensemble_output = {}

    for k in label_mapper.keys():
        ensemble_output[k] = pd.DataFrame(0.0, index=range(len(df_output["detectCls"][k])), columns=["fname", "fpath"] + [i for i in label_mapper[k].keys()], dtype="int32")
        
        # detection이 안 된 경우
        # detection & classification 아키텍처가 제대로 예측하지 못할 것이라고 판단하여 singleshot 아키텍처의 가중치를 많이 두고 앙상블
        prob_detect_n = (0.1 * df_output["detectCls"][k][(df_output["detectCls"][k]["x"] == -1).values].filter(regex="^prob_").replace(0.0, 0.333)) + \
            (0.8 * df_output["singleshot"][k][(df_output["detectCls"][k]["x"] == -1).values].filter(regex="^prob_")) + \
            (0.1 * df_output["detectCls_resnet"][k][(df_output["detectCls"][k]["x"] == -1).values].filter(regex="^prob_").replace(0.0, 0.333))
        cls_detect_n = {}
        for i in label_mapper[k].keys():
            cols = prob_detect_n.filter(regex=f"^prob_{i}").columns
            cls_detect_n[i] = prob_detect_n[cols].values.argmax(axis=1)

        # detection이 된 경우
        # 두 아키텍처를 가중평균하여 앙상블한 후, 위치 및 크기의 경우는 detection & classification 아키텍처의 값을 항상 사용
        prob_detect_y = (0.3 * df_output["detectCls"][k][(df_output["detectCls"][k]["x"] != -1).values].filter(regex="prob_*")) + \
            (0.5 * df_output["singleshot"][k][(df_output["detectCls"][k]["x"] != -1).values].filter(regex="prob_*")) + \
            (0.2 * df_output["detectCls_resnet"][k][(df_output["detectCls"][k]["x"] != -1).values].filter(regex="prob_*"))
        cls_detect_y = {}
        for i in label_mapper[k].keys():
            if i in ["size", "loc"]:
                cols = df_output["detectCls"][k].filter(regex=f"^cls_{i}").columns
                cls_detect_y[i] = df_output["detectCls"][k][(df_output["detectCls"][k]["x"] != -1).values][cols].values.flatten()
            else:
                cols = prob_detect_y.filter(regex=f"^prob_{i}").columns
                cls_detect_y[i] = prob_detect_y[cols].values.argmax(axis=1)

        ensemble_output[k]["fname"] = df_output["detectCls"][k]["fname"].values.copy()
        ensemble_output[k]["fpath"] = df_output["detectCls"][k]["fpath"].values.copy()
        for i in label_mapper[k].keys():
            ensemble_output[k].loc[(df_output["detectCls"][k]["x"] == -1).values, i] = cls_detect_n[i]
            ensemble_output[k].loc[(df_output["detectCls"][k]["x"] != -1).values, i] = cls_detect_y[i]

    for k in ensemble_output.keys():
        ensemble_output[k].to_csv(f"./ensemble_{k}_output.csv", index=False)


if __name__ == "__main__":
    main()
