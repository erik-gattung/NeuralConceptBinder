import os
import subprocess


def extract_frames(video_path, frames_target_base_path):
    os.makedirs(frames_target_base_path, exist_ok=True)
    # command = f"cd {os.path.dirname(video_path)} && ffmpeg -i {os.path.basename(video_path)} -q:v 2 -start_number 0 -loglevel quiet {frames_target_base_path}/'%05d.png'"
    command = f"cd {os.path.dirname(video_path)} && ffmpeg -i {os.path.basename(video_path)} -q:v 2 -start_number 0 -loglevel quiet {frames_target_base_path}/'%05d.jpg'"
    result = subprocess.run(command, shell=True)
    print(result)


def main():
    # declare video_id
    min_vid, max_vid = 10000, 11000
    video_ids = list(range(min_vid, max_vid))

    # setup paths based on video id
    video_paths = [f"/app/clevrer_videos/videos_val/video_{min_vid}-{max_vid}/video_{vid}.mp4" for vid in video_ids]
    frames_target_base_path = [f"/app/clevrer_videos/video_frames_val/video_{vid}" for vid in video_ids]

    [extract_frames(vid_path, target_path) for vid_path, target_path in zip(video_paths, frames_target_base_path)]

if __name__ == "__main__":
    main()