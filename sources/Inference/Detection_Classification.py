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
    batch_size = 4

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

# DNN모델로 cropping된 이미지를 분류하는 클래스 (각 타입별 모델을 따로 학습했기에 따로 예측)
class effinet_classifier():
    def __init__(self,
            model_dic={"house": None, "tree": None, "person": None},
            model_params={"house": {"num_classes": 7}, "tree": {"num_classes": 10}, "person": {"num_classes": 8} }
        ):
        # 모델 하이퍼파라미터 설정 및 전처리 함수 선언
        self.model_dic = model_dic
        self.model_params = model_params
        self.img_tranasformer = A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            A.Normalize(),
            ToTensorV2(),
        ])
    def infer(self, x={"house": None, "tree": None, "person": None}, y=None, batch_size=4, device="gpu"):
        # 결과물 저장 변수 초기화
        output_raw = {"house": None, "tree": None, "person": None}
        output_prob = {"house": None, "tree": None, "person": None}
        output_cls = {"house": None, "tree": None, "person": None}
        for i in ["house", "tree", "person"]:
            output_prob[i] = {}
            output_cls[i] = {}
            for j in label_mapper[i]:
                output_prob[i][j] = None
                output_cls[i][j] = None
        # torch device 설정
        if torch.cuda.is_available() and (device == "gpu"):
            infer_device = torch.device('cuda')
        else:
            infer_device = torch.device('cpu')

        for img_category in label_mapper.keys():
            # 위치 및 크기일 경우 예측하지 않도록 조건 지정 (규칙기반 알고리즘을 통해 예측함)
            if x[img_category] is None:
                continue
            # 모델 타입에 맞는 DNN torch 모델 로딩
            model = EfficientNet.from_pretrained('efficientnet-b4', **self.model_params[img_category])
            model.load_state_dict(torch.load(f"./models/crop/{img_category}_efficientnet.pth"))
            model.eval()
            # 이미지 전처리 함수 실행
            img_feature = torch.stack([self.preprocessing(i) for i in x[img_category]])
            # batch 단위로 DNN 모델 output 산출
            output_raw[img_category] = self.get_output_from_model(
                model, img_feature, None,
                batch_size=batch_size, infer_device=infer_device,
            )
            # raw output을 클래스로 변환하는 후처리 함수 실행
            self.postprocessing(output_raw, output_prob, output_cls, img_category)
        # 결과값 리턴
        return output_raw, output_prob, output_cls
    def preprocessing(self, img_path):
        # 이미지 로딩
        img = cv2.imread(img_path)
        # 이미지 채널 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 모델 인풋을 위한 이미지 데이터 변환 (리사이징 & 스케일링)
        img = self.img_tranasformer(image=img)["image"]
        return img
    def get_output_from_model(self, model, x, y=None, batch_size=4, infer_device=torch.device('cpu')):
        model.to(infer_device)
        # batch 단위로 output을 산출
        test_dl = DataLoader(x, batch_size=batch_size, shuffle=False)
        output = []
        for batch in test_dl:
            with torch.no_grad():
                output.append(model(batch.to(infer_device)).detach().cpu().numpy())
        model.to(torch.device('cpu'))
        # batch 단위로 산출된 output을 결합하여 리턴
        return np.concatenate(output)
    def postprocessing(self, output_raw, output_prob, output_cls, img_category):
        # 각 이미지 타입별로 output 갯수가 다르므로 조건을 이용해 분류
        if img_category == "house":
            tmp_output = [
                softmax(output_raw[img_category][:, 0:2], axis=1),
                softmax(output_raw[img_category][:, 2:4], axis=1),
                softmax(output_raw[img_category][:, 4:7], axis=1),
            ]
            for k, v  in zip([i for i in label_mapper[img_category].keys() if i not in ["loc", "size"]], tmp_output):
                output_prob[img_category][k] = v
                output_cls[img_category][k] = output_prob[img_category][k].argmax(axis=1)
        elif img_category == "tree":
            tmp_output = [
                softmax(output_raw[img_category][:, 0:2], axis=1),
                softmax(output_raw[img_category][:, 2:4], axis=1),
                softmax(output_raw[img_category][:, 4:6], axis=1),
                softmax(output_raw[img_category][:, 6:8], axis=1),
                softmax(output_raw[img_category][:, 8:10], axis=1),
            ]
            for k, v  in zip([i for i in label_mapper[img_category].keys() if i not in ["loc", "size"]], tmp_output):
                if k in ["loc", "size"]:
                    continue
                output_prob[img_category][k] = v
                output_cls[img_category][k] = v.argmax(axis=1)
        elif img_category == "person":
            tmp_output = [
                softmax(output_raw[img_category][:, 0:2], axis=1),
                softmax(output_raw[img_category][:, 2:4], axis=1),
                softmax(output_raw[img_category][:, 4:6], axis=1),
                softmax(output_raw[img_category][:, 6:8], axis=1),
            ]
            for k, v  in zip([i for i in label_mapper[img_category].keys() if i not in ["loc", "size"]], tmp_output):
                if k in ["loc", "size"]:
                    continue
                output_prob[img_category][k] = v
                output_cls[img_category][k] = v.argmax(axis=1).astype("int32")
        else:
            -1


# detection 좌표 output으로 위치 및 크기를 규칙기반 알고리즘으로 분류하는 클래스
class loc_size_classification:
    def loc(coordinate):
        if 0 < coordinate[0] < 2338.3:
            return 0
        elif 2338.3 <= coordinate[0] < 4676.6:
            return 1
        elif coordinate[0] >= 4676.6:
            return 2
        else :
            return 0

    def HT_size(coordinate):
        if (coordinate[2] != -1) & (coordinate[3] != -1):
            if 0 < coordinate[2]*coordinate[3] < 5567104:
                return 0
            elif 5567104 <= coordinate[2]*coordinate[3] < 20876640:
                return 1
            elif coordinate[2]*coordinate[3] >= 20876640:
                return 2
        else :
            return 0
        
    def P_size(coordinate):
        if (coordinate[2] != -1) & (coordinate[3] != -1):
            if 0 < coordinate[2]*coordinate[3] < 5567104:
                return 0
            elif 5567104 <= coordinate[2]*coordinate[3] < 13917760:
                return 1
            elif coordinate[2]*coordinate[3] >= 13917760:
                return 2     
        else :
            return 0

    def set_loc_size(test_df, loc_list, size_list):
        test_df['loc'] = loc_list
        test_df['size'] = size_list


def main(image_root_path=None):
    if image_root_path is None:
        # df_metainfo = get_metainfo_dataframe("./dataset/raw/val_images/", "./tmp/")
        parser = argparse.ArgumentParser(description='Classification with Detection Architecture')
        parser.add_argument('--image_root_path')
        args = parser.parse_args()
        image_root_path = args.image_root_path
    df_metainfo = get_metainfo_dataframe(image_root_path)

    if os.path.exists(CFG.detection_root_path):
        for img_category in label_mapper.keys():
            tmp = pickleIO(None, f"./detect_res_{img_category}.pkl", "r")
            df_metainfo[img_category][["x", "y", "w", "h"]] = tmp.values
    else:
        # Load the model
        model_detection = YOLO(f"./models/detection/yolov5mu_best.pt")
        # inference
        createFolder(CFG.detection_root_path)
        result_detect = {}
        for img_category in label_mapper.keys():
            result_detect[img_category] = []
            for batch in DataLoader(df_metainfo[img_category]["fpath"].values, batch_size=CFG.batch_size, shuffle=False):
                result_detect[img_category].extend(model_detection.predict([Image.open(i).resize((CFG.detection_img_size, CFG.detection_img_size)) for i in batch], imgsz=CFG.detection_img_size, conf=0.5, device="0"))
        del model_detection
        torch.cuda.empty_cache()
        gc.collect()
        
        for img_category in label_mapper.keys():
            output = []
            for img_name, img_path, img_res in zip(df_metainfo[img_category]["fname"], df_metainfo[img_category]["fpath"], result_detect[img_category]):
                if len(img_res) > 0:
                    tmp = pd.Series({i.boxes.conf.item(): i.boxes.xywh.detach().cpu().numpy().flatten() for i in img_res})
                    tmp = tmp.iloc[np.argmax([i[2] * i[3] for i in tmp.values])]
                    convert_and_crop_images(tmp, img_path, CFG.detection_root_path + img_name + ".jpg")
                    output.append(tmp)
                else:
                    shutil.copy(img_path, CFG.detection_root_path + img_name + ".jpg")
                    output.append(np.ones(4, dtype="float32") * -1.0)
            df_metainfo[img_category][["x", "y", "w", "h"]] = np.stack(output)
            pickleIO(df_metainfo[img_category][["x", "y", "w", "h"]], f"./detect_res_{img_category}.pkl", "w")
            
    model = effinet_classifier()
    inputs = {
        "house": (CFG.detection_root_path + df_metainfo["house"]["fname"].astype("str") + ".jpg").to_list(),
        "tree": (CFG.detection_root_path + df_metainfo["tree"]["fname"].astype("str") + ".jpg").to_list(),
        "person": (CFG.detection_root_path + df_metainfo["person"]["fname"].astype("str") + ".jpg").to_list(),
    }
    result_clf_raw, result_clf_prob, result_clf_cls = model.infer(inputs)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # loc & size를 제외한 classification 결과 저장를 dataframe에 저장 하는 프로세스
    for img_category in label_mapper.keys():
        # loc & size를 제외한 컬럼명을 추출합니다
        cols = df_metainfo[img_category].filter(regex="|".join([f"^raw_{i}" for i in label_mapper[img_category] if i not in ["loc", "size"]])).columns
        # DNN 모델의 raw output을 저장합니다
        df_metainfo[img_category][cols]= result_clf_raw[img_category]
        # raw output을 확률값으로 변환한 값을 저장합니다
        for k, v in result_clf_prob[img_category].items():
            if v is not None:
                cols = df_metainfo[img_category].filter(regex=f"^prob_{k}").columns
                df_metainfo[img_category][cols] = v
        # 최종 분류된 클래스 값을 저장합니다
        for k, v in result_clf_cls[img_category].items():
            if v is not None:
                cols = "cls_" + k
                df_metainfo[img_category][cols] = v
                
    # insert output to dataframe (undetected)
    for img_category in label_mapper.keys():
        df_coord = df_metainfo[img_category][["x", "y", "w", "h"]]
        if img_category == "person":
            res = [
                np.array([loc_size_classification.loc((x, y, w, h)) for x, y, w, h in zip(df_coord["x"], df_coord["y"], df_coord["w"], df_coord["h"])], dtype="int32"),
                np.array([loc_size_classification.P_size((x, y, w, h)) for x, y, w, h in zip(df_coord["x"], df_coord["y"], df_coord["w"], df_coord["h"])], dtype="int32"),
            ]   
        else:
            res = [
                np.array([loc_size_classification.loc((x, y, w, h)) for x, y, w, h in zip(df_coord["x"], df_coord["y"], df_coord["w"], df_coord["h"])], dtype="int32"),
                np.array([loc_size_classification.HT_size((x, y, w, h)) for x, y, w, h in zip(df_coord["x"], df_coord["y"], df_coord["w"], df_coord["h"])], dtype="int32"),
            ]
        for k, v in zip(df_metainfo[img_category].filter(regex="|".join([f"cls_{i}" for i in ["loc", "size"]])).columns, res):
            df_metainfo[img_category][k] = v

    for k in df_metainfo.keys():
        df_metainfo[k].to_csv(f"./detectCls_{k}_output.csv", index=False)

    return 0


if __name__ == "__main__":
    main()
