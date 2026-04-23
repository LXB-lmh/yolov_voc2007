from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")
model.predict(source=0, show=True, imgsz=640, conf=0.25, save=False)
