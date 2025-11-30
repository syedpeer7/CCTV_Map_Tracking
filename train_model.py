#model file - train
from ultralytics import YOLO
DATA_YAML = "/content/suspect_dataset/data.yaml"  # path created above
# choose a small model for fast training; change epochs as needed
model = YOLO("yolov8n.pt")  # pretrained base
# Train — this will use Colab GPU if available
model.train(data=DATA_YAML, epochs=30, imgsz=640, batch=8, name="suspect_train")
# After training, weights will be at runs/detect/suspect_train/weights/best.pt
WEIGHTS_PATH = "runs/detect/suspect_train/weights/best.pt"
print("Trained weights:", WEIGHTS_PATH)
