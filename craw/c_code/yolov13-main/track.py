import json
import time
import cv2
from collections import defaultdict
from ultralytics import YOLO

'''
def main(video_path, flush_every=2000, output_video="detection_result2.mp4"):
    model = YOLO(r"/home/nanchang/XW2/yolov13/main/yolov13-main/runs/ytb5122/weights/best.pt")

    frame_info   = defaultdict(list)
    trajectories = defaultdict(list)
    track_hits   = defaultdict(int)   
    prev_ids     = set()              
    tic  = time.perf_counter()
    frame_idx = 0

    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    results = model.track(
        source=video_path,
        stream=True,
        vid_stride=1,
        conf=0.5,                 
        iou=0.25,
        tracker='botsort.yaml',  
        persist=True,
        show=False,                 
        save=True,
        verbose=True,
        device='0'
    )

    for result in results:
        curr_ids = set()               
        if result.boxes and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            clses   = result.boxes.cls.int().cpu().tolist()

            for (x, y, w, h), tid, cls in zip(boxes, track_ids, clses):
                track_hits[tid] += 1
                if track_hits[tid] < 5:       
                    continue
                curr_ids.add(tid)

         
                frame_info[frame_idx].append({
                    "id": int(tid),
                    "cls": int(cls),
                    "cx": float(x),
                    "cy": float(y),
                    "w": float(w),
                    "h": float(h)
                })
                trajectories[tid].append((float(x), float(y)))
        for tid in (prev_ids - curr_ids):
            track_hits[tid] = max(0, track_hits[tid] - 1)
        prev_ids = curr_ids

        frame = result.plot()
        out.write(frame)

        frame_idx += 1
        if frame_idx % flush_every == 0:
            with open("xxx", "w") as f:
                json.dump(frame_info, f, indent=2)
            with open("xxx", "w") as f:
                json.dump(trajectories, f, indent=2)

    out.release()

    with open("xx", "w") as f:
        json.dump(frame_info, f, indent=2)
    with open("xx", "w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"{time.perf_counter()-tic:.2f}s")


if __name__ == "__main__":
    video_path = r"/home/nanchang/XW2/yolov13/main/video_beach/0_test.mp4"
    main(video_path, output_video="detection_result.mp4")
