from ultralytics import YOLO

model = YOLO("yolov8n.yaml") # build new model from scratch

results = model.train(data='config.yaml', epochs='1') #train the model






