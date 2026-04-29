from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="Objects365.yaml", 
    epochs=20, 
    imgsz=640, 
    device=0,
    batch=32,
    freeze=10, 
    optimizer="AdamW",
    weight_decay=0.0005,
    lr0=1e-3,
    project="runs",
    name="trained_objects",
    save=True
    )