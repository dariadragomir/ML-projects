from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

results = model.train(
    data='/Users/dariadragomir/Facultate/AN3/CAVA/proiect2/data.yaml', 
    epochs=50,              
    imgsz=480,                
    patience=5,
    batch=16,
    name='scooby_model'      
)