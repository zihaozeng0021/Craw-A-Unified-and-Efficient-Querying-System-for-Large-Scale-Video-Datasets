from __future__ import annotations
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import time

class Config:
    JSON_PATH = r'xxxxx'
    MIN_WINDOW_LENGTH = 80  
    MAX_WINDOW_LENGTH = 800 
    INIT_OVERLAP_THRESHOLD = 0.2  
    INIT_DISJOINT_RATIO = 0.3    
    MAX_EXPANSION_FACTOR = 10    
    EXPANSION_THRESHOLD = 0.4   
    STATS_BINS = 4            
    GUIDANCE_SMOOTH_FACTOR = 0.5  

Frame = int
TrackId = int
Range = Tuple[Frame, Frame]
durations = []

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)



def id_set_at_frame(data, frame):
    return {item['id'] for item in data.get(str(frame), [])}


def init(data):
    track2frames: defaultdict[TrackId, List[Frame]] = defaultdict(list)
    init_ids = id_set_at_frame(data, 0)
    total_init = len(init_ids) if init_ids else 1  

    disjoint_idx: Frame | None = None  
    tmp_idx: Frame | None = None       

    frames = sorted(int(k) for k in data.keys())
    max_data_frame = frames[-1] if frames else 0

    for frame in frames:  
        curr_ids = id_set_at_frame(data, frame)
        for tid in curr_ids:
            track2frames[tid].append(frame)


        if frame > 0 and total_init > 0:
            overlap_ratio = len(init_ids & curr_ids) / total_init


            if disjoint_idx is None and init_ids.isdisjoint(curr_ids):
                disjoint_idx = frame
            if tmp_idx is None and overlap_ratio <= Config.INIT_OVERLAP_THRESHOLD:
                tmp_idx = frame

            if disjoint_idx is not None and tmp_idx is not None:
                break


    init_duration = {
        tid: (frames[0], frames[-1])
        for tid, frames in track2frames.items()
    }
    durations.append(init_duration)
    

    if first_break_candidate := (tmp_idx or disjoint_idx or max_data_frame):
        if first_break_candidate - 0 + 1 < Config.MIN_WINDOW_LENGTH:
            first_break_candidate = min(0 + Config.MIN_WINDOW_LENGTH - 1, max_data_frame)
    

    if disjoint_idx is not None and tmp_idx is not None:
        if (disjoint_idx - tmp_idx) / tmp_idx > Config.INIT_DISJOINT_RATIO:
 
            return tmp_idx if tmp_idx - 0 + 1 >= Config.MIN_WINDOW_LENGTH else \
                   min(tmp_idx + (Config.MIN_WINDOW_LENGTH - (tmp_idx - 0 + 1)), max_data_frame)
        return disjoint_idx if disjoint_idx - 0 + 1 >= Config.MIN_WINDOW_LENGTH else \
               min(disjoint_idx + (Config.MIN_WINDOW_LENGTH - (disjoint_idx - 0 + 1)), max_data_frame)
    return first_break_candidate


def build_track_duration_map(data, start, end):
    track2frames: defaultdict[TrackId, List[Frame]] = defaultdict(list)
    
    for frame in range(start, end + 1):
        frame_str = str(frame)
        if frame_str in data:  
            for item in data[frame_str]:
                track2frames[item['id']].append(frame)

    track_duration = {}
    for tid, frames in track2frames.items():
        if frames: 
            first_frame = min(frames)  
            last_frame = max(frames)   
            track_duration[tid] = (first_frame, last_frame)
    
    return track_duration


def check_stats(track_durations):
    if not track_durations:
        return {}
        

    duration_list = [last - first for first, last in track_durations.values()]
    if not duration_list:  
        return {}
        
    min_dur = min(duration_list)
    max_dur = max(duration_list)

    if min_dur == max_dur:
        return {(min_dur, max_dur): len(duration_list)}
        

    bin_width = (max_dur - min_dur) / Config.STATS_BINS
    bins_dict = {}
    

    for i in range(Config.STATS_BINS):
        bin_start = min_dur + i * bin_width
        bin_end = min_dur + (i + 1) * bin_width
        
 
        if all(isinstance(d, int) for d in duration_list):
            bin_start = int(round(bin_start))
            bin_end = int(round(bin_end))
            
            if i == Config.STATS_BINS - 1:
                bin_end = max_dur
        
        bins_dict[(bin_start, bin_end)] = 0
    

    for dur in duration_list:
        assigned = False
        for i, (start, end) in enumerate(bins_dict.keys()):
     
            if (i == 0 and dur >= start and dur <= end) or \
               (i > 0 and dur > start and dur <= end):
                bins_dict[(start, end)] += 1
                assigned = True
                break

        if not assigned:
            closest_bin = min(bins_dict.keys(), 
                             key=lambda x: abs((x[0] + x[1])/2 - dur))
            bins_dict[closest_bin] += 1
    
    return {k: v for k, v in bins_dict.items() if v > 0}


def dynamic_split(data, first_break):
    max_frame = max(int(k) for k in data.keys())
    msu_ranges: List[Range] = []
    msu_ranges.append((0, first_break))
    
    initial_duration = build_track_duration_map(data, 0, first_break)
    if initial_duration:
        init_hist = check_stats(initial_duration)
        if init_hist:
            mode_bin = max(init_hist.items(), key=lambda x: x[1])[0]
         
            guidance_length = max(int((mode_bin[0] + mode_bin[1]) / 2), Config.MIN_WINDOW_LENGTH)
    else:
        guidance_length = max(first_break + 1, Config.MIN_WINDOW_LENGTH)
    
    current = first_break + 1
    msu_index = 1
    
    while current <= max_frame:
        msu_index += 1
        remaining_frames = max_frame - current + 1 
    
        expansion_factor = 1
        window_len = guidance_length * expansion_factor
        window_len = min(window_len, remaining_frames, Config.MAX_WINDOW_LENGTH)
        window_len = max(window_len, Config.MIN_WINDOW_LENGTH)
        end = current + window_len - 1
        expansion_attempts = 0
        max_attempts = Config.MAX_EXPANSION_FACTOR
        expanded = False
        
        while expansion_attempts < max_attempts:
            duration = build_track_duration_map(data, current, end)
            if not duration:
                break 
                
            hist = check_stats(duration)
            bin_items = sorted(hist.items(), key=lambda x: x[0])
            counts = [count for (_, _), count in bin_items]
            max_count = max(counts) if counts else 0
            total_objects = sum(counts)
            
            if total_objects == 0:
                break
                
            current_ratio = max_count / total_objects
            max_bin = max(hist.items(), key=lambda x: x[1])[0]
            max_bin_length = max_bin[1] - max_bin[0]
            
            if current_ratio > Config.EXPANSION_THRESHOLD and expansion_factor < Config.MAX_EXPANSION_FACTOR:
                new_expansion_factor = expansion_factor + 1
                new_window_len = guidance_length * new_expansion_factor
                new_window_len = max(new_window_len, max_bin_length)
                new_window_len = min(new_window_len, remaining_frames, Config.MAX_WINDOW_LENGTH)
                new_end = current + new_window_len - 1
                
                if new_end > end and new_end <= max_frame:
                    expanded = True
                    expansion_factor = new_expansion_factor
                    expansion_attempts += 1
                    end = new_end
                    window_len = new_window_len
                else:
                    break
            else:
                break

        if not expanded:
            current_duration = build_track_duration_map(data, current, end)
            if current_duration:
                current_hist = check_stats(current_duration)
                max_bin = max(current_hist.items(), key=lambda x: x[1])[0]
                max_bin_length = max_bin[1] - max_bin[0]
                
                adjusted_length = max_bin_length if max_bin_length >= Config.MIN_WINDOW_LENGTH else Config.MIN_WINDOW_LENGTH
                adjusted_length = min(adjusted_length, Config.MAX_WINDOW_LENGTH, remaining_frames)
                
                if adjusted_length != window_len:
                    end = current + adjusted_length - 1
                    window_len = adjusted_length
        

        final_window_length = end - current + 1
        if final_window_length < Config.MIN_WINDOW_LENGTH:
            new_end = current + Config.MIN_WINDOW_LENGTH - 1
            end = new_end if new_end <= max_frame else max_frame
        elif final_window_length > Config.MAX_WINDOW_LENGTH:
            end = current + Config.MAX_WINDOW_LENGTH - 1
        
        msu_ranges.append((current, end))
        final_duration = build_track_duration_map(data, current, end)
        durations.append(final_duration)
        
        final_hist = check_stats(final_duration)
        if final_hist:
            mode_bin = max(final_hist.items(), key=lambda x: x[1])[0]
            new_guidance = int((mode_bin[0] + mode_bin[1]) / 2)
            guidance_length = max(
                int(guidance_length * Config.GUIDANCE_SMOOTH_FACTOR + new_guidance * (1 - Config.GUIDANCE_SMOOTH_FACTOR)),
                Config.MIN_WINDOW_LENGTH
            )
        
        current = end + 1

    return msu_ranges, durations


def main():
    time1 = time.perf_counter()
    data = load_json(Config.JSON_PATH)
    
    first_break = init(data)
    msu_list, durations = dynamic_split(data, first_break)
    time2 = time.perf_counter()


    #print(f"\n总分割数: {len(durations)}")
    return msu_list, durations


if __name__ == '__main__':
    msu_list, durations = main()