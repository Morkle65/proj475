from ultralytics import YOLO

model = YOLO("runs/trained_objects_2/weights/best.pt")

model.train(
    task="classify"
    data="data/places365",
    epochs=50,
    imgsz=640,
    batch=64,
    lr0=0.0001,
    freeze=0,
    optimizer="Adam",
    device=0,
    )
