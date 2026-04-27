from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="Objects365.yaml", 
    epochs=30, 
    imgsz=640, 
    device=0,
    batch=32,
    freeze=10, 
    optimizer="Adam", 
    lr0=1e-3,
    project="runs",
    name="trained_objects_1",
    save=True
    )

model = YOLO("runs/objects_train/weights/best.pt")

model.train(
    data="Objects365.yaml",
    epochs=70,
    imgsz=640,
    device=0,
    batch=32,
    freeze=10,
    optimizer="Adam",
    lr0=1e-4,
    project="runs",
    name="trained_objects_2",
    save=True
    )