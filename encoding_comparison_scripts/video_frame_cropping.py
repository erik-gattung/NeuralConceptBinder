# TODO: Crop video frame into 128x128 images based on derender proposals if possible

import numpy as np
from pycocotools import mask as maskUtils
import os
import json
from PIL import Image


# Decode RLE binary masks and get furthest points left and right where masks are active
def get_left_right_col_index(mask_list: list[dict]):
    left = np.inf
    right = -1

    for rle_mask in mask_list:
        binary_mask = maskUtils.decode(rle_mask)

        cols_with_mask = np.where(binary_mask.max(axis=0) > 0)[0]

        if cols_with_mask.size > 0:  # Ensure mask is not empty
            leftmost_x = cols_with_mask.min()  # Leftmost pixel
            rightmost_x = cols_with_mask.max()  # Rightmost pixel

            if leftmost_x < left:
                left = leftmost_x
            if rightmost_x > right:
                right = rightmost_x

    mask_width = right - left + 1
    img_height, img_width = binary_mask.shape

    if mask_width <= img_height:
        center = (left + right) // 2  # Find the center of the mask
        half_size = img_height // 2  # Half of the square size

        crop_col_1 = max(0, center - half_size)
        crop_col_2 = min(img_width, center + half_size)

        # Ensure the crop size is exactly img_height
        if crop_col_2 - crop_col_1 < img_height:
            crop_col_2 = min(img_width, crop_col_1 + img_height)
            crop_col_1 = max(0, crop_col_2 - img_height)

        return (crop_col_1, crop_col_2)  # take image[:, crop_x1:crop_x2]
    else:
        return None


def main():
    proposal_dir = "/app/clevrer_videos/derender_proposals"

    proposal_files = sorted(os.listdir(proposal_dir))[10000:15000]
    proposal_files = [os.path.join(proposal_dir, pf) for pf in proposal_files]

    video_frame_root_dir = "/app/clevrer_videos/video_frames_val"
    square_frames_root_dir = "/app/clevrer_videos/square_frames_val"

    for pf in proposal_files:
        with open(pf, "r") as f:
            proposal_dict = json.load(f)

        video_index = proposal_dict["video_index"]

        for frame_dict in proposal_dict["frames"]:
            frame_index = frame_dict["frame_index"]
            masks = [obj["mask"] for obj in frame_dict["objects"]]
            image_croppings = get_left_right_col_index(masks)

            if image_croppings is not None:
                image = Image.open(
                    os.path.join(
                        video_frame_root_dir,
                        "video_" + str(video_index),
                        str(frame_index).zfill(5) + ".png",
                    )
                )

                image = image.crop(
                    (image_croppings[0], 0, image_croppings[1], image.height)
                ).resize((128, 128), Image.LANCZOS)

                image.save(
                    os.path.join(
                        square_frames_root_dir,
                        f"video_{video_index}_frame_{str(frame_index).zfill(3)}.png",
                    )
                )


if __name__ == "__main__":
    main()
