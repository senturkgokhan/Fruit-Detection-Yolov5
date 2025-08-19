import cv2
import torch
import numpy as np
import pathlib
from pathlib import Path
import sys

# === PosixPath hatasını engelle ===
pathlib.PosixPath = pathlib.WindowsPath

# === yolov5 klasörünü projeye dahil et ===
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolov5'  # yolov5 klasörü aynı dizinde olmalı
sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# === scale_coords fonksiyonu (manuel) ===
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x
    coords[:, [1, 3]] -= pad[1]  # y
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    return coords

# === Model ayarları ===
weights = r"C:\Users\sentu\OneDrive\Desktop\FruitDetectionYoloV5\yolov5\best.pt"
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights, device=device)
model.eval()
names = model.names
img_size = 640
conf_thres = 0.5
iou_thres = 0.45

# === Kamerayı başlat ===
cap = cv2.VideoCapture(0)  # 0 = varsayılan kamera
if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img0 = frame.copy()
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

    detected_count = 0
    if pred[0] is not None and len(pred[0]):
        pred[0][:, :4] = scale_coords(img_tensor.shape[2:], pred[0][:, :4], img0.shape).round()

        for *xyxy, conf, cls in pred[0]:
            label = f'{names[int(cls)]} {conf:.2f}'
            cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            detected_count += 1

    cv2.putText(img0, f"Tespit Sayisi: {detected_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Kamera - YOLOv5", img0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
