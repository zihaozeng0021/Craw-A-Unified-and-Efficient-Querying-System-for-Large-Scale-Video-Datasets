import json
import os
from collections import defaultdict

def stat(frame_info_path, output_file=None):

    with open(frame_info_path, 'r') as f:
        frame_info = json.load(f)

    object_frames = defaultdict(list)
    

    for frame_id, objects in frame_info.items():
        frame_num = int(frame_id)
        

        for obj in objects:
            obj_id = obj["id"]
            object_frames[obj_id].append(frame_num)
    
    for obj_id in object_frames:
        object_frames[obj_id].sort()
    
    ids = 0
    framess = 0
    
    for obj_id, frames in sorted(object_frames.items()):
        tmp = frames[-1] - frames[0]
        ids += 1
        framess += tmp
    

    if output_file:
        
        result_dict = {str(k): v for k, v in object_frames.items()}
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    return object_frames

if __name__ == "__main__":
    frame_info_path = "xxx"
    output_file = "xx"
    stat(frame_info_path, output_file)
