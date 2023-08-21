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
    img_size = 384
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
    def __init__(self, feature, label, loss_type="binary"):
        self.feature = torch.from_numpy(feature).to(torch.float32)
        self.label = torch.from_numpy(label).to(torch.float32 if loss_type == "binary" else torch.int64)
        
    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return {"feature": self.feature[idx], "label": self.label[idx]}

class DNN_CustomModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Dropout(),
            nn.Linear(params["n_input"], params["hidden_layers"]),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(params["hidden_layers"], params["hidden_layers"] // 4),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(params["hidden_layers"] // 4, params["n_classes"])

    def forward(self, feature):
        feature = self.lin(feature)
        feature = self.classifier(feature)
        return feature

class DNN_TreeClassifier():
    def __init__(self, model_params, model_threshold_params, model_backbone, feature_extractor):
        self.model_list = []
        self.model_threshold_spliter = []
        self.eval_score_list = []
        self.threshold_eval_score_list = []
        self.n_classes = model_params["n_classes"]
        self.model_backbone = model_backbone
        self.feature_extractor = feature_extractor
        self.criterion = nn.BCEWithLogitsLoss() if model_params["n_classes"] == 1 else nn.CrossEntropyLoss()
        for idx, params in enumerate(product(*model_threshold_params["dynamic_params"].values())):
            tmp_params = model_threshold_params["fixed_params"].copy()
            tmp_params["random_state"] = idx
            tmp_params.update({k: v for k, v in zip(model_threshold_params["dynamic_params"].keys(), params)})
            self.model_threshold_spliter.append(DecisionTreeClassifier(**tmp_params))
        self.model_list.append(DNN_CustomModel(model_params))
    def fit(self, x, y, eval_x, eval_y):
        train_dl = DataLoader(CustomDataset(x, y, loss_type="binary" if self.n_classes == 1 else "multiclass"), shuffle=True, batch_size=CFG.batch_size, drop_last=True)
        valid_dl = DataLoader(CustomDataset(eval_x, eval_y, loss_type="binary" if self.n_classes == 1 else "multiclass"), shuffle=False, batch_size=CFG.batch_size)
        print(f"INFO: number of iteration : {len(train_dl)}\n")
        for model in self.model_list:
            model.to(device)
            model.train()
            do_training(model, train_dl, valid_dl, self.criterion)
            model.eval()
            y_prob = self.do_infer(model, eval_x)
            self.eval_score_list.append(skl_metrics.log_loss(eval_y, y_prob))
        x_pred = self.do_infer(self.model_list[np.argmin(self.eval_score_list)], x)
        eval_x_pred = self.do_infer(self.model_list[np.argmin(self.eval_score_list)], eval_x)
        for model in self.model_threshold_spliter:
            model.fit(x_pred, y)
            y_prob = model.predict_proba(eval_x_pred)
            self.threshold_eval_score_list.append(skl_metrics.f1_score(eval_y, y_prob.argmax(axis=1), average="macro"))
    def predict_proba(self, x):
        output = {"y_prob": self.do_infer(self.model_list[np.argmin(self.eval_score_list)], x)}
        output["y_prob_threshold_opt"] = self.model_threshold_spliter[np.argmax(self.threshold_eval_score_list)].predict_proba(output["y_prob"])
        return output
    def do_infer(self, model, x, force_to_cpu=False):
        model.eval()
        model.to(torch.device("cpu") if force_to_cpu else device)
        test_dl = DataLoader(CustomDataset(x, np.zeros(len(x)), loss_type="binary" if self.n_classes == 1 else "multiclass"), shuffle=False, batch_size=CFG.batch_size)
        y_pred = []
        for batch in test_dl:
            with torch.no_grad():
                feature = self.model_backbone(self.feature_extractor(batch["feature"], return_tensors="pt")["pixel_values"].to(torch.device("cpu") if force_to_cpu else device)).pooler_output
            with torch.no_grad():
                output = model(feature)
            if output.shape[-1] == 1:
                output = output.sigmoid()
                output = torch.cat([1 - output, output], dim=-1)      
            else:
                output = F.softmax(output, dim=-1)
            y_pred.append(output.detach().cpu().numpy())
        model.to(torch.device("cpu"))
        return np.concatenate(y_pred)
    
class swin_classifier():
    def __init__(self):
        self.resizer = transforms.Resize((CFG.img_size, CFG.img_size))
    def infer(self, x={"house": None, "tree": None, "person": None}, use_dt_thresholder=True):
        # 결과물 저장 변수 초기화
        # output_raw = {"house": None, "tree": None, "person": None}
        output_prob = {"house": None, "tree": None, "person": None}
        output_cls = {"house": None, "tree": None, "person": None}
        for i in ["house", "tree", "person"]:
            output_prob[i] = {}
            output_cls[i] = {}
            for j in label_mapper[i]:
                output_prob[i][j] = None
                output_cls[i][j] = None

        for img_category in label_mapper.keys():
            # 모델 로딩
            model = pickleIO(None, f"./models/singleshot/{img_category}_swinV1.pkl", "r")
            # 추론
            img_feature = np.stack([self.img_preprocessor(i) for i in x[img_category]])
            for target in label_mapper[img_category]:
                output = [i.predict_proba(img_feature) for i in model[target]]
                output_prob[img_category][target] = np.stack([i["y_prob"] for i in output], axis=0).mean(axis=0)
                if use_dt_thresholder:
                    output_cls[img_category][target] = np.stack([i["y_prob_threshold_opt"] for i in output], axis=0).mean(axis=0).argmax(axis=1)
                else:
                    output_cls[img_category][target] = output_prob[img_category][target].argmax(axis=1)
        return output_prob, output_cls
    def img_preprocessor(self, fpath):
        img = read_image(fpath)
        img = self.resizer(img)
        return img
    
def main(image_root_path=None):
    if image_root_path is None:
        # df_metainfo = get_metainfo_dataframe("./dataset/raw/val_images/", "./tmp/")
        parser = argparse.ArgumentParser(description='Classification with Detection Architecture')
        parser.add_argument('--image_root_path')
        args = parser.parse_args()
        image_root_path = args.image_root_path
    df_metainfo = get_metainfo_dataframe(image_root_path)

    inputs = {
        "house": df_metainfo["house"]["fpath"].to_list(),
        "tree": df_metainfo["tree"]["fpath"].to_list(),
        "person": df_metainfo["person"]["fpath"].to_list(),
    }
    model = swin_classifier()
    result_clf_prob, result_clf_cls = model.infer(inputs)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # loc & size를 제외한 classification 결과 저장를 dataframe에 저장 하는 프로세스
    for img_category in label_mapper.keys():
        # loc & size를 제외한 컬럼명을 추출합니다
        cols = df_metainfo[img_category].filter(regex="|".join([f"^raw_{i}" for i in label_mapper[img_category]])).columns
        # DNN 모델의 raw output을 저장합니다
        df_metainfo[img_category][cols]= -1.0
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

    for k in df_metainfo.keys():
        df_metainfo[k].to_csv(f"./singleshot_{k}_output.csv", index=False)

    return 0

if __name__ == "__main__":
    main()

