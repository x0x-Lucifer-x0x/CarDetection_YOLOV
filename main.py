from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")  # build a new model from scratch

# Use the model
model.train(data="config.yaml",batch=8, imgsz=640, epochs=1,workers=1)  # train the model
