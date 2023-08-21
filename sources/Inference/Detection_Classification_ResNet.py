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
import argparse
import shutil

from sklearn import linear_model as lm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics as skl_metrics

import cv2
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
        'door_yn':{'n':0, 'y':1}, # �� ����
        'loc':{'left':0, 'center':1, 'right':2}, # ��ġ
        'roof_yn':{'y':1, 'n':0}, # ���� ����
        'window_cnt':{'absence':0, '1 or 2':1, 'more than 3':2}, # â�� ����
        'size':{'small':0, 'middle':1, 'big':2}, # ũ��
    },
    "tree": {
        "branch_yn": {"n": 0, "y": 1}, # ���� ����
        "root_yn": {"n": 0, "y": 1}, # �Ѹ� ����
        "crown_yn": {"n": 0, "y": 1}, # ���� ����
        "fruit_yn": {"n": 0, "y": 1}, # ���� ����
        "gnarl_yn": {"n": 0, "y": 1}, # ���̳���ó ����
        "loc": {"left": 0, "center": 1, "right": 2}, # ��ġ
        "size": {"small": 0, "middle": 1, "big": 2}, # ũ��
    },
    "person": {
        "eye_yn": {"n": 0, "y": 1}, # �� ����
        "leg_yn": {"n": 0, "y": 1}, # �ٸ� ����
        "loc": {"left": 0, "center": 1, "right": 2}, # ��ġ
        "mouth_yn": {"n": 0, "y": 1}, # �� ����
        "size": {"small": 0, "middle": 1, "big": 2}, # ũ��
        "arm_yn": {"n": 0, "y": 1}, # �� ����
    }
}

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

class CustomDataset(Dataset):
    def __init__(self, feature, label=None):
        self.feature = torch.from_numpy(feature).to(torch.float32)
        self.label = torch.ones(len(self.feature), dtype=torch.int64) if label is None else torch.from_numpy(label).to(torch.int64)
        
    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return {"feature": self.feature[idx], "label": self.label[idx]}

class DNN_ResNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, params["n_output"])
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class resnet_classifier():
    def __init__(self):
        self.img_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((CFG.img_size, CFG.img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def infer(self, x={"house": None, "tree": None, "person": None}):
        # ����� ���� ���� �ʱ�ȭ
        output_raw = {"house": None, "tree": None, "person": None}
        output_prob = {"house": None, "tree": None, "person": None}
        output_cls = {"house": None, "tree": None, "person": None}
        for i in ["house", "tree", "person"]:
            output_raw[i] = {}
            output_prob[i] = {}
            output_cls[i] = {}
            for j in label_mapper[i]:
                output_raw[i] = {}
                output_prob[i][j] = None
                output_cls[i][j] = None
        for img_category in label_mapper.keys():
            img_feature = np.stack([self.img_preprocessor(i) for i in x[img_category]])
            test_dl = DataLoader(CustomDataset(img_feature, None), batch_size=CFG.batch_size, shuffle=False)
            for target in label_mapper[img_category]:
                if target in ["loc", "size"]:
                    continue
                output = {}
                model = DNN_ResNet({"n_output": len(label_mapper[img_category][target].keys())})
                model.load_state_dict(torch.load(f"./models/singleshot_resnet/{img_category}/ResNet101_{target}.pt"))
                model.eval()
                model.to(device)
                output = self.get_output_from_model(model, test_dl)
                output_raw[img_category][target] = output["raw"]
                output_prob[img_category][target] = output["prob"]
                output_cls[img_category][target] = output["prob"].argmax(axis=1)
        return output_raw, output_prob, output_cls    
    def img_preprocessor(self, fpath):
        img = Image.open(fpath)
        img = self.img_transformer(img)
        return img
    def get_output_from_model(self, model, test_dl):
        output = []
        for batch in test_dl:
            with torch.no_grad():
                output.append(model(batch["feature"].to(device)))
        output = torch.cat(output, dim=0)
        return {"raw": output.detach().cpu().numpy(), "prob": F.softmax(output, dim=1).detach().cpu().numpy()}
    
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

    inputs = {
        "house": (CFG.detection_root_path + df_metainfo["house"]["fname"].astype("str") + ".jpg").to_list(),
        "tree": (CFG.detection_root_path + df_metainfo["tree"]["fname"].astype("str") + ".jpg").to_list(),
        "person": (CFG.detection_root_path + df_metainfo["person"]["fname"].astype("str") + ".jpg").to_list(),
    }
    model = resnet_classifier()
    result_clf_raw, result_clf_prob, result_clf_cls = model.infer(inputs)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # loc & size�� ������ classification ��� ���带 dataframe�� ���� �ϴ� ���μ���
    for img_category in label_mapper.keys():
        # loc & size�� ������ �÷����� �����մϴ�
        cols = df_metainfo[img_category].filter(regex="|".join([f"^raw_{i}" for i in label_mapper[img_category] if i not in ["loc", "size"]])).columns
        # DNN ���� raw output�� �����մϴ�
        df_metainfo[img_category][cols]= -1.0
        # raw output�� Ȯ�������� ��ȯ�� ���� �����մϴ�
        for k, v in result_clf_prob[img_category].items():
            if v is not None:
                cols = df_metainfo[img_category].filter(regex=f"^prob_{k}").columns
                df_metainfo[img_category][cols] = v
        # ���� �з��� Ŭ���� ���� �����մϴ�
        for k, v in result_clf_cls[img_category].items():
            if v is not None:
                cols = "cls_" + k
                df_metainfo[img_category][cols] = v

    for k in df_metainfo.keys():
        df_metainfo[k].to_csv(f"./detectCls_resnet_{k}_output.csv", index=False)

    return 0

if __name__ == "__main__":
    main()
