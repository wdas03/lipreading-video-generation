import os
import glob
import pickle

from decord import VideoReader
from decord import cpu

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import sys

DATA_DIR = "/home/whd2108/mvlrs_v1"

def get_mp4_files(directory):
    # Use a glob pattern to match all .mp4 files
    pattern = os.path.join(directory, '**', '*.mp4')
    mp4_files = glob.glob(pattern, recursive=True)
    return mp4_files

def process_video(video_path):
    try:
        # Initialize the VideoReader with the CPU context
        vr = VideoReader(video_path, ctx=cpu(0))  # Use CPU context for reading videos
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)

        if video_fps == 0:
            raise ValueError("FPS is zero, which may indicate an issue with the video file or codec.")

        # Calculate the step size to simulate an effective FPS of 30 if needed
        step = max(1, int(video_fps / 30))
        frame_indices = [(i, i + step) for i in range(0, total_frames - step, step)]

        return video_path, frame_indices

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return video_path, []

def main(mp4_files):
    video_dataset_by_frame = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_video = {executor.submit(process_video, mp4): mp4 for mp4 in mp4_files}
        for future in tqdm(as_completed(future_to_video), total=len(mp4_files), desc="Processing videos"):
            video_path, frame_indices = future.result()
            video_dataset_by_frame[video_path] = frame_indices

    return video_dataset_by_frame

class FrameItem:
    def __init__(self, video_path, frame_start, frame_end):
        self.video_path = video_path
        self.frame_start = frame_start
        self.frame_end = frame_end

def create_frame_item(args):
    file_path, start_end = args
    start, end = start_end
    return FrameItem(file_path, start, end)

def process_data(data):
    tasks = [(file_path, (start, end)) for file_path, indices in data.items() for (start, end) in indices]

    all_instances = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all tasks to the executor
        future_to_item = {executor.submit(create_frame_item, task): task for task in tasks}

        # Initialize tqdm progress bar
        progress = tqdm(total=len(future_to_item), desc="Processing Frames", file=sys.stdout)

        # As futures complete, update progress bar
        for future in as_completed(future_to_item):
            result = future.result()  # Get the result of the completed future
            all_instances.append(result)
            progress.update(1)  # Update progress bar by one each time a future completes
            progress.refresh()  # Force refresh the output

        progress.close()  # Ensure to close the progress bar after completion

    return all_instances

def extract_instances(data):
    all_instances = []
    for file_path, indices in tqdm(data.items()):
        for (start, end) in indices:
            all_instances.append(FrameItem(file_path, start, end))

    return all_instances

if __name__ == "__main__":
    if not os.path.exists("../lrs_video_files.pkl"):
        mp4_files = get_mp4_files(DATA_DIR)
        pickle.dump(mp4_files, open("../lrs_video_files.pkl", "wb+"))
    else:
        mp4_files = pickle.load(open("../lrs_video_files.pkl", "rb"))

    assert len(mp4_files) > 0

    if os.path.exists("video_dataset_by_frame.pkl"):
        print("Extracting individual dataset instances...")
        # all_instances = process_data(pickle.load(open("video_dataset_by_frame.pkl", "rb")))
        all_instances = extract_instances(pickle.load(open("video_dataset_by_frame.pkl", "rb")))
        print(len(all_instances))

        pickle.dump(all_instances, open("video_dataset_list.pkl", "wb+"))
    else:
        print("Extracting files...")
        video_dataset_by_frame = main(mp4_files)
        pickle.dump(video_dataset_by_frame, open("video_dataset_by_frame.pkl", "wb+"))
