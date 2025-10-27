
import ffmpeg
import cv2
from video_split import main
from collections import defaultdict
from typing import Dict, List, Tuple

def get_video_time_by_frame(video_path, msu_list):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # right FPS
    if fps <= 0:
        fps = 30.0 
    
    for idx, (s, e) in enumerate(msu_list, start=1):
        frame_start = s
        frame_end = e            
        start_time = frame_start / fps
        end_time = frame_end / fps
        
        (
            ffmpeg.input(video_path)
            .filter('trim', start=start_time, end=end_time)
            .filter('setpts', 'PTS-STARTPTS')  
            .output(
                f"xxx",
                vsync=2, 
                reset_timestamps=1  
            )
            .overwrite_output() 
            .run()
        )


if __name__ == "__main__":
    msu_list, durations = main()
    video_path = r'/home/nanchang/ZZQ/yolov13/main/video_youtobe/youtobe1.mp4'
    get_video_time_by_frame(video_path, msu_list)


