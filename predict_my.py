from ultralytics import YOLO

model = YOLO('weights/best.pt')
model.predict(
    source=0,          # 0=默认摄像头，1=第二个摄像头
    imgsz=640,         # 和训练一致先用640
    conf=0.35,         # 建议先0.3~0.4，减少误检
    iou=0.45,          # NMS阈值，默认可用
    show=True,         # 实时弹窗显示
    save=False,        # 不保存视频
    device=0,          # 有GPU就0；没GPU可删掉
    stream=False,     # 流式推理更稳
    verbose=False
)