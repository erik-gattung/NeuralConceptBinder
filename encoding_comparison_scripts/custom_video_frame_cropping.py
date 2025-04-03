# TODO: Crop video frame into 128x128 images based on derender proposals if possible

import numpy as np
import os
from PIL import Image


def main():
    # video_frame_root_dir = "/app/custom_clevrer/blender_images/"
    # square_frames_root_dir = "/app/custom_clevrer/square_images"
    video_frame_root_dir = "/app/custom_clevrer/face_frames/"
    square_frames_root_dir = "/app/custom_clevrer/square_face_images"

    image_subfolder = [os.path.join(video_frame_root_dir, f) for f in os.listdir(video_frame_root_dir) if os.path.isdir(os.path.join(video_frame_root_dir, f))]
    for vf in image_subfolder:
        video_number = os.path.basename(vf).split("_")[1]
        image_files = sorted([os.path.join(vf, f) for f in os.listdir(vf) if f.lower().endswith((".png", ".jpeg", ".jpg"))])

        for image_fp in image_files:
            image_number = int(os.path.basename(image_fp).split("_")[2].split(".")[0])
            img = np.array(Image.open(image_fp))
            H, W = img.shape[:2]
            half_x, half_y = H//2, W//2
            cropped_img = Image.fromarray(img[:, half_y - half_x: half_y + half_x, :])

            cropped_img = cropped_img.resize((128, 128), Image.LANCZOS)

            cropped_img.save(
                    os.path.join(
                        square_frames_root_dir,
                        f"video_{video_number}_frame_{str(image_number).zfill(3)}.png",
                    )
                )
            

if __name__ == "__main__":
    main()
