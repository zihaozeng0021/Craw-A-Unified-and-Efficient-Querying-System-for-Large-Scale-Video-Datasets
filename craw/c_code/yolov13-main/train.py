import warnings, os,logging
os.environ["YOLO_OFFLINE"] = "1" 
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"     
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'/home/nanchang/ZZQ/yolov13/main/yolov13-main/ultralytics/cfg/models/v13/yolov13.yaml')
    model.load(r'/home/nanchang/ZZQ/yolov13/main/yolov13-main/yolov13l.pt')
    model.train(data=r'/home/nanchang/ZZQ/yolov13/main/yolov13-main/ultralytics/cfg/data3.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=32,
                workers=4,
                device='0,1,2,3',
                # resume='', # last.pt path
                name='ytb6',
                )

