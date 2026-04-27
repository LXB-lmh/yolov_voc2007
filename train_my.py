
from ultralytics import YOLO


def main() -> None:
    # 4090 24G + ~10k dataset high-performance training baseline
    model = YOLO("weights/yolov8l.pt")
    model.train(
        data="../VOCdevkit/YOLO_Dataset/dataset.yaml",      # TODO: replace with your dataset yaml path
        device=0,              # single GPU (4090)
        workers=12,            # tune in [8, 16] based on CPU cores
        imgsz=640,
        batch=-1,              # auto batch size (fills VRAM safely)
        epochs=300,
        cache="ram",           # if RAM not enough, change to "disk"
        amp=True,              # mixed precision for faster training
        optimizer="SGD",       # usually better final accuracy for detection
        cos_lr=True,
        close_mosaic=10,
        patience=50,
        save=True,
        save_period=10,
        pretrained=True,
        deterministic=False,   # max throughput; set True only for strict reproducibility
        project="runs/train",
        name="yolov8l_10k_4090",
    )


if __name__ == "__main__":
    main()