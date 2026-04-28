from ultralytics import YOLO

model = YOLO("weights/best.pt")
model.predict(
    source=0,
    imgsz=640,
    conf=0.5,
    iou=0.5,
    device=0,
    show=True,
    save=False,
)
