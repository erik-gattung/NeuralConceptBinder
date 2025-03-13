# -*- coding: utf-8 -*-
"""
Created on Fri February 07 11:35:57 2025

@author: Erik Gattung
@organization: DATAbility GmbH
@copyright: DATAbility GmbH
@contact: gattung@datability.ai
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
import os
import shutil
from skimage.measure import label, regionprops


def plot_image_seg_mask(image: Image, segmentation_mask: np.ndarray):
    """
    Visualize the segmentation mask and its image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    # Plot the original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")  # Hide the axes
    # Plot the segmentation mask
    axes[1].imshow(segmentation_mask, cmap="gnuplot")  # Adjust colormap as needed
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")  # Hide the axes
    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()


def plot_image_flow_seg_mask(
    image: Image, flow_img: Image, tresh_flow: np.ndarray, seg_masks: list[np.ndarray]
):
    """
    Visualize the original image, the optical flow, its tresholded norm and the segmentation mask.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    # Plot the original image
    axes[0][0].imshow(image)
    axes[0][0].set_title("Original video frame")
    axes[0][0].axis("off")  # Hide the axes

    # Plot the flow viz image
    axes[0][1].imshow(flow_img, cmap="grey")
    axes[0][1].set_title("Optical Flow Visualization")
    axes[0][1].axis("off")  # Hide the axes

    # Plot the segmentation mask
    labeled_array = np.zeros_like(seg_masks[0], dtype=int)
    for idx, mask in enumerate(seg_masks, start=1):
        labeled_array[mask] = idx

    axes[1][0].imshow(labeled_array, cmap="gnuplot")  # Adjust colormap as needed
    axes[1][0].set_title("Segmentation Masks")
    axes[1][0].axis("off")  # Hide the axes

    # Plot the thresholded flow array
    axes[1][1].imshow(tresh_flow, cmap="grey")
    axes[1][1].set_title("Thresholded flow array")
    axes[1][1].axis("off")  # Hide the axes

    plt.tight_layout()
    plt.show()


def save_segmentation_mask(seg_mask: np.ndarray, fp: str):
    """
    Save the segmentation mask as a greyscale image
    """
    Image.fromarray(seg_mask.astype(np.uint8)).convert("L").save(fp)


def convert_seg_mask_to_coherent_region_masks(seg_mask: np.ndarray):
    """
    From a greyscale segmentation mask, extract connected regions of the same color,
    at least of 0.5% of the total image size and create a binary mask for each region.
    Returns list of binary region masks
    """
    labeled_array, num_features = label(seg_mask)
    region_masks = []
    for region_id in range(1, num_features + 1):
        # Create a binary mask where pixels belonging to this region are 1
        mask = (labeled_array == region_id).astype(np.uint8)
        if (
            np.sum(mask) <= 0.005 * seg_mask.size
        ):  # mask active on at least 0.5% of the image
            continue
        region_masks.append(mask)

    return region_masks


def plot_connected_regions(regions: list[np.ndarray]):
    """
    Plot connected regions in a 3xN plot.
    """
    num_regions = len(regions)
    cols = 3  # Number of columns
    rows = (num_regions + cols - 1) // cols  # Compute number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    axes = axes.flatten()
    plt.suptitle("Connected, distinct regions")

    for i, region in enumerate(regions):
        axes[i].imshow(region, cmap="gray")
        axes[i].set_title(f"Region {i}")
        axes[i].axis("off")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def intersect_flow_seg(
    region_masks: list[np.ndarray],
    tresh_flow: np.ndarray,
    bounding_box_bloat=35,
    intersection_treshold=0.5,
):
    """
    Intersect regions masks from segmentation with tresholded optical flow to find moving objects.
    Returns masks and bloated bounding boxes of moving objects.
    """
    moving_object_masks = []
    bounding_boxes = []

    for region in region_masks:
        region_intersection = np.sum(region * tresh_flow) / np.sum(region)
        if region_intersection > intersection_treshold:
            moving_object_masks.append(region)

            rows, cols = np.where(region == 1)

            # Step 2: Compute the bounding box
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            min_row_inflated = max(
                min_row - bounding_box_bloat, 0
            )  # Ensure top boundary doesn't go below 0
            max_row_inflated = min(
                max_row + bounding_box_bloat, region.shape[0] - 1
            )  # Ensure bottom boundary doesn't exceed height
            min_col_inflated = max(
                min_col - bounding_box_bloat, 0
            )  # Ensure left boundary doesn't go below 0
            max_col_inflated = min(
                max_col + bounding_box_bloat, region.shape[1] - 1
            )  # Ensure right boundary doesn't exceed width

            # The bounding box is defined by (min_row, min_col) as the top-left corner and (max_row, max_col) as the bottom-right corner
            bounding_boxes.append(
                (min_row_inflated, min_col_inflated, max_row_inflated, max_col_inflated)
            )

    return moving_object_masks, bounding_boxes


def plot_image_bounding_box_and_mask(image, region_masks, bounding_boxes):
    """
    Plots the original image, superimposes the bounding box and visualizes the region mask
    multiplied by the image.
    """
    if len(region_masks) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))  # Create a figure and an axis
        ax.imshow(image)  # Display the image on the axis
        fig.suptitle(
            f"Image {os.path.basename(image.filename)} with no moving masks"
        )  # Set figure title
        ax.axis("off")  # Hide axis
        plt.show()
        return

    n_rows = len(region_masks)

    fig, axes = plt.subplots(n_rows, 2, figsize=(6, 2 * n_rows))
    plt.suptitle(f"Moving image patches for Image {os.path.basename(image.filename)}")

    # If only one row, make axes a 2D list for consistency
    if n_rows == 1:
        axes = np.array([axes])

    for i, (moving, box) in enumerate(zip(region_masks, bounding_boxes)):
        if i >= n_rows:
            break  # Stop if more regions exist than allocated rows

        rect = plt.Rectangle(
            (box[1], box[0]),
            box[3] - box[1],
            box[2] - box[0],
            linewidth=2,
            edgecolor="black",
            facecolor="red",
            alpha=0.25,
        )

        # Plot the original video frame with bounding box
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Video Frame")
        axes[i, 0].add_patch(rect)
        axes[i, 0].axis("off")

        # Plot mask filled with video frame
        masked_image = image * moving[:, :, np.newaxis]
        axes[i, 1].imshow(masked_image)
        axes[i, 1].set_title(f"Video Frame * Region Mask")
        axes[i, 1].axis("off")

    plt.tight_layout()  # Adjust layout to fit title
    plt.show()


def mask_center(mask):
    """
    Computes the center x, y coords of the mask.
    """
    # Get the coordinates of the True pixels
    y, x = np.where(mask)

    # Compute the centroid
    if len(y) == 0 or len(x) == 0:
        return None  # Return None if the mask is empty

    center_x = np.mean(x)
    center_y = np.mean(y)

    return (center_x, center_y)


def reorder_masks_compute_mask_centers(masks, boxes):
    """
    SAM2 has a different ordering than plt for boxes. Rearrange box info to match SAM2
    conversions. Also computes mask centers.
    Returns boxes to fit SAM2 conversions and mask centers.
    """
    for i in range(len(boxes)):
        boxes[i] = [boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]]

    mask_centers = [mask_center(mask) for mask in masks]

    return boxes, mask_centers


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax, color: str = "green"):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2)
    )


def save_segmented_image(
    img_folder: str, img_name: str, mask: np.ndarray, target_folder: str
):
    img = np.array(Image.open(os.path.join(img_folder, img_name)))
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w)
    img[~mask] = 0
    Image.fromarray(img).save(os.path.join(target_folder, img_name))
    return img


def load_selected_frames(video_dir, start_frame, end_frame):
    frame_files = sorted(
        [f for f in os.listdir(video_dir) if f.endswith(".jpg") or f.endswith(".png")]
    )
    selected_frames = frame_files[start_frame:end_frame]
    frames = []
    for frame_file in selected_frames:
        frame_path = os.path.join(video_dir, frame_file)
        frame = Image.open(frame_path)
        frames.append(np.array(frame))
    return np.array(frames)


def clean_and_copy_images(destination_folder: str, image_paths: list[str]):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Remove all existing files in the destination folder
    for filename in os.listdir(destination_folder):
        file_path = os.path.join(destination_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    copied_paths = list()

    # Copy image files to the destination folder
    for image_path in image_paths:
        if os.path.isfile(image_path):  # Ensure the file exists
            try:
                img = Image.open(image_path).convert("RGB").resize((384, 256))
                copy_path = os.path.join(destination_folder, os.path.basename(image_path))
                img.save(copy_path)
                copied_paths.append(copy_path)
            except Exception as e:
                print(f"Failed to process {image_path}. Reason: {e}")

    return copied_paths


def plot_sam2_masks_boxes(image, boxes, image_index, out_mask_logits, out_obj_ids):
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {image_index}")
    plt.imshow(image)
    for i in range(len(boxes)):
        color = np.array([*cmap(out_obj_ids[i][i])[:3], 1])
        show_box(boxes[i], plt.gca(), color=color)
        show_mask(
            (out_mask_logits[i][i] > 0.0).cpu().numpy(),
            plt.gca(),
            obj_id=out_obj_ids[i][i],
        )


def plot_images(images: list[str], vis_frame_stride=10):
    num_images = len(range(0, len(images), vis_frame_stride))
    cols = 4  # Fixing 4 columns
    rows = int(np.ceil(num_images / cols))  # Calculating required rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()  # Flatten axes for easy indexing

    for idx, out_frame_idx in enumerate(range(0, len(images), vis_frame_stride)):
        ax = axes[idx]
        ax.set_title(f"Frame {images[out_frame_idx].split('/')[-1]}")
        ax.imshow(Image.open(images[out_frame_idx]))
        ax.axis("off")  # Hide axis for cleaner visualization

    # Hide any unused subplots
    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_sam2_masks_propagation(images: list[str], video_segments, vis_frame_stride=10):
    num_images = len(range(0, len(images), vis_frame_stride))
    cols = 4  # Fixing 4 columns
    rows = int(np.ceil(num_images / cols))  # Calculating required rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()  # Flatten axes for easy indexing

    for idx, out_frame_idx in enumerate(range(0, len(images), vis_frame_stride)):
        ax = axes[idx]
        ax.set_title(f"Frame {images[out_frame_idx].split('/')[-1]}")
        ax.imshow(Image.open(images[out_frame_idx]))

        if out_frame_idx in video_segments.keys():
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask > 0, ax, obj_id=out_obj_id)
        else:
            ax.set_title(
                f"Frame {images[out_frame_idx].split('/')[-1]} - No mask found"
            )

        ax.axis("off")  # Hide axis for cleaner visualization

    # Hide any unused subplots
    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_image_mask_from_array(images: list[str], masks: np.ndarray):
    num_images = len(range(0, len(images)))
    cols = 4  # Fixing 4 columns
    rows = int(np.ceil(num_images / cols))  # Calculating required rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()  # Flatten axes for easy indexing

    for idx in range(0, len(images)):
        ax = axes[idx]
        ax.set_title(f"Frame {images[idx].split('/')[-1]}")
        ax.imshow(Image.open(images[idx]))

        for obj_id in range(len(masks)):
            show_mask(masks[obj_id][idx], ax, obj_id=obj_id)

        # if idx % 2 == 0:
        #     ax.set_title(f"Original masks")
        # else:
        #     ax.set_title(f"Warped masks")

        ax.axis("off")  # Hide axis for cleaner visualization

    # Hide any unused subplots
    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def save_image_mask_from_array(images: list[str], masks: np.ndarray, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

    for idx, image_path in enumerate(images):
        fig, ax = plt.subplots(figsize=(6, 4))  # Create a single subplot per image
        img = Image.open(image_path)

        ax.imshow(img)

        # Overlay masks
        for obj_id in range(len(masks)):
            show_mask(masks[obj_id][idx], ax, obj_id=obj_id)

        ax.axis("off")  # Hide axis for cleaner visualization

        # Save the result
        image_num = int(image_path.split("/")[-1].split(".")[0])
        save_path = os.path.join(save_dir, f"segmented_{image_num:04d}.jpg")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)  # Close figure to free memory


def show_anns(image, anns, borders=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

    plt.axis("off")
    plt.show()


def get_fitted_sampling_points(tresh_flow: np.ndarray, points_per_side: int = 32):
    """
    Given `tresh_flow`, compute the sampling points that are within the mask and return
    them as a Nx2 array. Index 0 along width, index 1 along height.
    """
    y_coords = np.linspace(0, 1, points_per_side)
    x_coords = np.linspace(0, 1, points_per_side)
    # Generate the grid using meshgrid
    xx, yy = np.meshgrid(x_coords, y_coords)
    # Stack and reshape the points into (N, 2) format
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    H, W = tresh_flow.shape

    # Convert normalized coordinates to pixel indices
    pixel_coords = (points * np.array([H, W])).astype(int)
    pixel_coords = np.clip(pixel_coords, 0, [H - 1, W - 1])  # Ensure within bounds

    # Keep only points where mask value is 1
    valid_mask = tresh_flow[pixel_coords[:, 0], pixel_coords[:, 1]] > 0
    filtered_points = points[valid_mask]

    # swap columns to match contract
    return filtered_points[:, [1, 0]]


def remove_outliers_from_masks(seg_masks: list[np.ndarray]):
    filtered_masks = []

    for mask in seg_masks:
        # Label the connected components in the mask
        labeled_mask = label(mask)

        # Get properties of the labeled regions
        regions = regionprops(labeled_mask)

        # Find the largest region (based on area)
        largest_region = max(regions, key=lambda r: r.area)
        largest_region_label = largest_region.label

        # Create a new mask that only keeps the largest region
        new_mask = (labeled_mask == largest_region_label).astype(np.uint8)

        # Append the filtered mask
        filtered_masks.append(new_mask)

    return filtered_masks


def filter_min_size(seg_masks: list[np.ndarray], size_treshold: int = 400):
    filtered_masks = []

    for mask in seg_masks:
        if mask.sum() > size_treshold:
            filtered_masks.append(mask)

    return filtered_masks


def iou(mask1: np.ndarray, mask2: np.ndarray):
    """
    Compute the Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0.0


def filter_masks_by_iou(masks: list[np.ndarray], iou_threshold=0.5):
    """
    Filters the list of masks based on their IoU values. Keeps only the smaller
    mask in the pair if IoU > threshold.
    """

    # Calculate the IoU matrix and apply the filtering rule
    num_masks = len(masks)
    iou_matrix = -1 * np.ones((num_masks, num_masks))  # init matrix with -1

    for i in range(num_masks):
        for j in range(i + 1, num_masks):
            # Compute IoU between mask i and mask j
            iou_score = iou(masks[i], masks[j])
            iou_matrix[i, j] = iou_score

    # Iterate over the IoU matrix and remove masks based on IoU > threshold
    masks_to_remove = set()
    for i in range(num_masks):
        for j in range(i + 1, num_masks):
            if iou_matrix[i, j] > iou_threshold:
                # Keep the smaller mask
                if np.sum(masks[i]) > np.sum(masks[j]):
                    masks_to_remove.add(i)
                else:
                    masks_to_remove.add(j)

    # Filter out the larger masks based on the IoU threshold rule
    return [
        mask.astype(bool) for i, mask in enumerate(masks) if i not in masks_to_remove
    ]


def post_process_masks(
    seg_masks: list[np.ndarray], iou_treshold: int = 0.5, size_treshold: int = 400
):
    """
    Main function to process the list of binary masks.
    It filters out outliers and removes large masks based on IoU score.
    """
    # Step 1: Remove outliers connected to the largest region
    filtered_masks = remove_outliers_from_masks(seg_masks)

    # Step 2: Remove masks below treshold size
    filtered_masks = filter_min_size(filtered_masks, size_treshold=size_treshold)

    # Step 3: Filter based on IoU threshold
    filtered_masks = filter_masks_by_iou(filtered_masks, iou_threshold=iou_treshold)

    return filtered_masks
