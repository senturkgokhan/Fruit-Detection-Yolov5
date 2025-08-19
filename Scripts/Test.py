import subprocess

# YOLOv5 detect.py komutunu çalıştır
subprocess.run([
    "python", "detect.py",
    "--weights", r"C:\Users\sentu\OneDrive\Desktop\FruitDetectionYoloV5\yolov5\best.pt",
    "--img", "640",
    "--conf-thres", "0.25",
    "--source", r"C:\Users\sentu\OneDrive\Desktop\mixed_2.jpg",
    "--save-txt",
    "--save-conf"
])
