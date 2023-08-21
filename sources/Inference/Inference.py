import os
import argparse
from time import time
import pandas as pd
from datetime import datetime
import shutil

class CFG:
    detection_root_path = "./tmp/"

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

def main():
    total_runtime = time()

    time_container = {}
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image_root_path')
    args = parser.parse_args()
    image_root_path = args.image_root_path
    num_images = len(os.listdir(image_root_path))

    print("=== Detection ===")
    start_time = time()
    os.system(f"python Only_Detection.py --image_root_path {image_root_path}")
    time_container["Only_Detection"] = round(time() - start_time, 3)
    time_container["Only_Detection_Per_Image"] = round(time_container['Only_Detection'] / num_images, 3)
    print(f"Only_Detection time : {time_container['Only_Detection']} (per image : {time_container['Only_Detection_Per_Image']})")

    print("=== Detection_Classification ===")
    start_time = time()
    os.system(f"python Detection_Classification.py --image_root_path {image_root_path}")
    time_container["Detection_Classification"] = round(time() - start_time, 3)
    time_container["Detection_Classification_Per_Image"] = round(time_container['Detection_Classification'] / num_images, 3)
    print(f"Detection_Classification time : {time_container['Detection_Classification']} (per image : {time_container['Detection_Classification_Per_Image']})")

    print("=== Detection_Classification_ResNet ===")
    start_time = time()
    os.system(f"python Detection_Classification_ResNet.py --image_root_path {image_root_path}")
    time_container["Detection_Classification_ResNet"] = round(time() - start_time, 3)
    time_container["Detection_Classification_ResNet_Per_Image"] = round(time_container['Detection_Classification_ResNet'] / num_images, 3)
    print(f"Detection_Classification_ResNet time : {time_container['Detection_Classification_ResNet']} (per image : {time_container['Detection_Classification_ResNet_Per_Image']})")

    print("=== SingleShot_Classification ===")
    start_time = time()
    os.system(f"python SingleShot_Classification.py --image_root_path {image_root_path}")
    time_container["SingleShot_Classification"] = round(time() - start_time, 3)
    time_container["SingleShot_Classification_Per_Image"] = round(time_container['SingleShot_Classification'] / num_images, 3)
    print(f"SingleShot_Classification time : {time_container['SingleShot_Classification']} (per image : {time_container['SingleShot_Classification_Per_Image']})")

    print("=== Ensemble ===")
    start_time = time()
    os.system(f"python Ensemble.py")
    time_container["Ensemble"] = round(time() - start_time, 3)
    time_container["Ensemble_Per_Image"] = round(time_container['Ensemble'] / num_images, 3)
    print(f"Ensemble time : {time_container['Ensemble']} (per image : {time_container['Ensemble_Per_Image']})")

    time_container["TOTAL_RUNTIME"] = round(time() - total_runtime, 3)
    time_container["TOTAL_RUNTIME_Per_Image"] = round(time_container['TOTAL_RUNTIME'] / num_images, 3)
    print(f"TOTAL_RUNTIME time : {time_container['TOTAL_RUNTIME']} (per image : {time_container['TOTAL_RUNTIME_Per_Image']})")

    # save data
    datetime.now().strftime("log_%Y-%m-%d-%H-%M-%S")
    time_container = pd.Series(time_container)
    time_container.index.name = "process"
    time_container.name = "time(sec)"
    time_container.to_csv(f'./log_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv')
    
    # delete tmp files
    shutil.rmtree(CFG.detection_root_path)
    for img_category in label_mapper.keys():
        try:
            os.remove(f"./detect_res_{img_category}.pkl")
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    main()
