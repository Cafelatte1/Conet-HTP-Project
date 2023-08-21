GLOBAL_SEED = 42

import os
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

from PIL import Image
import numpy as np # linear algebra
from numpy import random as np_rnd
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import random as rnd
import pickle
import gc
import time
import shutil
import cv2
import argparse
from itertools import product

from sklearn import linear_model as lm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics as skl_metrics

from PIL import Image
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel
from torchvision import transforms
from torchvision.io import read_image

from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

from efficientnet_pytorch import EfficientNet
from scipy.special import softmax
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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
    n_folds = 5
    img_size = 224
    detection_img_size = 640
    detection_root_path = "./tmp/"
    channels = 3
    batch_size = 16


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

# func: 추론할 이미지에 대한 메타정보가 포함된 데이터프레임을 전처리하는 함수
def get_metainfo_dataframe(img_root_path):
    # get image path data
    df_metainfo = pd.DataFrame()
    df_metainfo["fname"] = sorted(os.listdir(img_root_path))
    df_metainfo["fname"] = df_metainfo["fname"].apply(lambda x: x.split(".")[0])
    df_metainfo["fpath"] = img_root_path + df_metainfo["fname"].astype("str") + ".jpg"
    df_metainfo["img_category"] = df_metainfo["fname"].apply(lambda x: x.split("_")[-1])
    df_metainfo = {k: df_metainfo.loc[df_metainfo["img_category"] == k].reset_index(drop=True) for k in ["house", "person", "tree"]}
    # initialize column values
    for img_category in label_mapper.keys():
        output_cols = []
        for i in label_mapper[img_category].keys():
            for j in label_mapper[img_category][i].keys():
                output_cols.append(i + "_" + j)
        df_metainfo[img_category][["raw_" + i for i in output_cols]] = 0.0
        df_metainfo[img_category][["raw_" + i for i in output_cols]] = df_metainfo[img_category][["raw_" + i for i in output_cols]].astype("float32")
        df_metainfo[img_category][["prob_" + i for i in output_cols]] = 0.0
        df_metainfo[img_category][["prob_" + i for i in output_cols]] = df_metainfo[img_category][["prob_" + i for i in output_cols]].astype("float32")
        df_metainfo[img_category][["cls_" + i for i in label_mapper[img_category].keys()]] = -1
        df_metainfo[img_category][["cls_" + i for i in label_mapper[img_category].keys()]] = df_metainfo[img_category][["cls_" + i for i in label_mapper[img_category].keys()]].astype("int32")
    return df_metainfo

# func: yolo 좌표게를 cv2 좌표계로 변환한 후 이미지를 cropping하여 저장하는 함수
def convert_and_crop_images(coord, import_path, export_path):
    x_min = int(coord[0]) - int(coord[2] / 2)
    x_max = int(coord[0]) + int(coord[2] / 2)
    y_min = int(coord[1]) - int(coord[3] / 2)
    y_max = int(coord[1]) + int(coord[3] / 2)
    img = cv2.imread(import_path)
    img = img[y_min:y_max, x_min:x_max]
    cv2.imwrite(export_path, img)


def main(image_root_path=None):
    if image_root_path is None:
        # df_metainfo = get_metainfo_dataframe("./dataset/raw/val_images/", "./tmp/")
        parser = argparse.ArgumentParser(description='Classification with Detection Architecture')
        parser.add_argument('--image_root_path')
        args = parser.parse_args()
        image_root_path = args.image_root_path
    df_metainfo = get_metainfo_dataframe(image_root_path)

    # Load the model
    model_detection = YOLO(f"./models/detection/yolov5mu_best.pt")
    # inference
    createFolder(CFG.detection_root_path)
    result_detect = {}
    for img_category in label_mapper.keys():
        result_detect[img_category] = {"conf": [], "coord": []}
        for batch in DataLoader(df_metainfo[img_category]["fpath"].values, batch_size=CFG.batch_size, shuffle=False):
            for output in model_detection.predict(batch, imgsz=(CFG.detection_img_size, CFG.detection_img_size), conf=0.5, device="0", half=True):
                result_detect[img_category]["conf"].append(output.boxes.conf.detach().cpu().numpy())
                result_detect[img_category]["coord"].append(output.boxes.xywh.detach().cpu().numpy())
            del output
            gc.collect()
            torch.cuda.empty_cache()

    del model_detection
    gc.collect()
    torch.cuda.empty_cache()
              
    # 추론 결과물을 활용해 raw이미지를 cropping하고 임시폴더에 저장합니다
    # 여러 bounding box가 검출 시, 가장 큰 bounding box를 선택합니다 - select 알고리즘은 추후 고도화 가능
    for img_category in label_mapper.keys():
        output = []
        for img_name, img_path, img_conf, img_coord in zip(df_metainfo[img_category]["fname"], df_metainfo[img_category]["fpath"], result_detect[img_category]["conf"], result_detect[img_category]["coord"]):
            # 검출된 객체가 하나 이상이면 가장 bounding box 넓이가 큰 것을 저장합니다
            if len(img_conf) > 0:
                # tmp = pd.Series({i.boxes.conf.item(): i.boxes.xywh.detach().cpu().numpy().flatten() for i in img_res})
                tmp = pd.Series({i: j for i, j in zip(img_conf, img_coord)})
                tmp = tmp.iloc[np.argmax([i[2] * i[3] for i in tmp.values])]
                convert_and_crop_images(tmp, img_path, CFG.detection_root_path + img_name + ".jpg")
                output.append(tmp)
            # 검출된 객체가 없으면 raw image를 그대로 저장합니다 (좌표값은 -1로 저장)
            else:
                shutil.copy(img_path, CFG.detection_root_path + img_name + ".jpg")
                output.append(np.ones(4, dtype="float32") * -1.0)
        df_metainfo[img_category][["x", "y", "w", "h"]] = np.stack(output)
        pickleIO(df_metainfo[img_category][["x", "y", "w", "h"]], f"./detect_res_{img_category}.pkl", "w")

    return 0


if __name__ == "__main__":
    main()
