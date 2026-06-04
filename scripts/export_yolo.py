import os
import subprocess
import sys

def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

def export_model():
    from ultralytics import YOLO
    
    # Load the YOLOv8 nano model (downloads automatically if missing)
    print("Downloading and loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    
    # Export the model to ONNX format, explicitly for dynamic input sizes or fixed 640x640
    print("Exporting model to ONNX...")
    path = model.export(format="onnx", imgsz=640, opset=12)
    
    # Move it to the models directory
    if os.path.exists("models/yolov8n.onnx"):
        os.remove("models/yolov8n.onnx")
    os.rename(path, "models/yolov8n.onnx")
    print(f"Model successfully exported and moved to models/yolov8n.onnx")

if __name__ == "__main__":
    install_requirements()
    export_model()
