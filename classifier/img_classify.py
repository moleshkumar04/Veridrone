from ultralytics import YOLO
import os

# Load a model
#model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
model_path = 'best.pt'

# Load the YOLOv8 model
model = YOLO(model_path)
names = model.names
image_files = []
def predict(image_name):
    image_path = f"uploads/{image_name}"
    results = model(image_path)

    # Extract information from the results
    number_of_persons = 0
    missing_hardhat = 'No'
    missing_ppe = 'No'
    
    for result in results:
        boxes = result.boxes  # Assuming detection
        for c in result.boxes.cls:
            print(names[int(c)])
            lbl = names[int(c)]
            if lbl == "person" :
                number_of_persons += 1 
            if lbl == "no-hardhat" :
                missing_hardhat = 'Yes'
            if lbl == "no-safetyvest" :
                missing_ppe = 'Yes'

        # Save the result image
        filename=f"static/uploads/{image_name}_result.jpg"
        result.save(filename)

    return number_of_persons, missing_hardhat, missing_ppe, filename
