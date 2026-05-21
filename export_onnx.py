from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(
    format="onnx",
    opset=12,  # broadly compatible; bump to 17 if a later TRT step complains
    dynamic=False,  # fixed input shape — simpler and faster
    simplify=True,  # runs onnx-simplifier; cleans up the graph
    imgsz=640,
)
print("Exported to yolov8n.onnx")
