
import ffmpeg
import os

def extract_frames(video_path, output_folder,
                   start_time="00:00:05",   
                   duration="00:00:10",     
                   fps=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    (
        ffmpeg
        .input(video_path, ss=start_time, t=duration)  
        .filter('fps', fps=fps)
        .output(os.path.join(output_folder, 'frame_%04d.png'),
                start_number=1)
        .run(overwrite_output=True)
    )

extract_frames(
    video_path=r'/home/nanchang/ZZQ/yolov13/main/video_Bellevue/Bellevue_116th_NE12th_2_test.mp4',
    output_folder=r'/home/nanchang/ZZQ/yolov13/main/xxx',
    start_time="00:00:00",   
    duration="00:00:01",     
    fps=1      
)