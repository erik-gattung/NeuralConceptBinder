import os
import subprocess
import shutil


def extract_frames(video_path, frames_target_base_path):
    os.makedirs(frames_target_base_path, exist_ok=True)
    # command = f"cd {os.path.dirname(video_path)} && ffmpeg -i {os.path.basename(video_path)} -q:v 2 -start_number 0 -loglevel quiet {frames_target_base_path}/'%05d.png'"
    command = f"cd {os.path.dirname(video_path)} && ffmpeg -i {os.path.basename(video_path)} -q:v 2 -start_number 0 -loglevel error {frames_target_base_path}/'%06d.jpg'"
    result = subprocess.run(command, shell=True)
    # print(result)


def move_all_frames_to_same_folder(video_folders, target_folder):
    os.makedirs(target_folder, exist_ok=True)

    for folder in video_folders:
        for filename in os.listdir(folder):
            source_path = os.path.join(folder, filename)
            new_filename = f"{os.path.basename(folder)}_{filename}"
            target_path = os.path.join(target_folder, new_filename)
            shutil.move(source_path, target_path)

        os.rmdir(folder)


def main():
    # declare video_id
    # min_vid, max_vid = 10000, 11000
    min_vid, max_vid = 1, 4 # only take the first 3 videos
    video_ids = list(range(min_vid, max_vid))

    # setup paths based on video id
    # video_paths = [f"/app/clevrer_videos/videos_val/video_{min_vid}-{max_vid}/video_{vid}.mp4" for vid in video_ids]
    # frames_target_base_path = [f"/app/clevrer_videos/video_frames_val/video_{vid}" for vid in video_ids]

    # TODO: Update folder names for Hessian AI cluster
    video_paths = [f"/app/UT_ego/UTE_videos/P0{vid}.mp4" for vid in video_ids]
    temp_img_base_path = [f"/app/UT_ego/video_frames/video_{vid}" for vid in video_ids]    # store them all in the same folder
    final_image_path = "/app/UT_ego/video_frames/"

    [extract_frames(vid_path, target_path) for vid_path, target_path in zip(video_paths, temp_img_base_path)]
    move_all_frames_to_same_folder(temp_img_base_path, final_image_path)


if __name__ == "__main__":
    main()
