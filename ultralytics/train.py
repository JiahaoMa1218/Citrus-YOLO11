import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model_name = 'citrus_yolo11.yaml'  
    
    data_yaml = '/root/autodl-tmp/ultralytics/orange.yaml'
    
    batch_size = 24      
    img_size = 640       
    total_epochs = 1000  
    
    project_name = 'citrus_project'
    exp_name = 'citrus_yolo11_run' 

    print(f"🚀 Starting training for custom model: {model_name}")
    print(f"⚙️  Config: Batch={batch_size} | Epochs={total_epochs} | Img_size={img_size}")
    
    model = YOLO(model_name)
    model.info()

    model.train(
        data=data_yaml,
        epochs=total_epochs,
        imgsz=img_size,
        batch=batch_size,
        device=0,
        workers=8,
        project=project_name,
        name=exp_name,
        amp=True,       
        exist_ok=True,  
        val=True,       
        patience=0,     
        save=True,      
        save_period=-1  
    )
