# -*- coding: utf-8 -*-
"""
Created on Mon February 17 17:06:06 2025

@author: Erik Gattung
@organization: DATAbility GmbH
@copyright: DATAbility GmbH
@contact: gattung@datability.ai
"""
import os
import sys
from PIL import Image
import numpy as np
import argparse
import joblib
import traceback
import sam2_pipeline_util as pod_u
from scipy.ndimage import gaussian_filter1d

os.chdir("/app/SEA-RAFT")
if "/app/RAFT/core" in sys.path:
    sys.path.remove("/app/RAFT/core")
if "/app/SEA-RAFT/core" not in sys.path:
    sys.path.append("/app/SEA-RAFT/core")
from sea_raft_demo import parse_args, create_model, estimate_flow

os.chdir("/app/sam2")
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def prepare_models():
    # init optical flow model
    sea_raft_parser = argparse.ArgumentParser()
    sea_raft_parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    sea_raft_parser.add_argument(
        "--path", help="checkpoint path", type=str, default=None
    )
    sea_raft_parser.add_argument("--url", help="checkpoint url", type=str, default=None)
    sea_raft_parser.add_argument(
        "--device", help="inference device", type=str, default="cpu"
    )
    args_list = [
        "--path",
        "/app/SEA-RAFT/models/SEA-RAFT-Tartan-C-T-TSKH-spring540x960-M.pth",
        "--cfg",
        "/app/SEA-RAFT/config/eval/spring-M.json",
        "--device",
        "cuda",
    ]
    sea_raft_args = parse_args(sea_raft_parser, args_list)
    sea_raft_model = create_model(sea_raft_args)

    # init segmentation model
    sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    device = "cuda"
    sam2_predictor = build_sam2_video_predictor(
        model_cfg, sam2_checkpoint, device=device
    )

    return (
        sea_raft_model,
        sea_raft_args,
        sam2_predictor,
    )


def estimate_flow_on_images(sea_raft_model, sea_raft_args, image_filenames):
    # call flow model
    flows, flow_vizs, heatmaps, heatmap_vizs = zip(
        *[
            estimate_flow(
                sea_raft_model,
                sea_raft_args,
                image_filenames[i],
                image_filenames[i + 1],
                device="cuda",
            )
            for i in range(len(image_filenames) - 1)
        ]
    )
    # Convert tensors to NumPy arrays
    flows = [flow.squeeze().cpu().numpy().transpose((1, 2, 0)) for flow in flows]
    flow_vizs = list(flow_vizs)
    heatmaps = [heatmap.squeeze().cpu().numpy() for heatmap in heatmaps]
    heatmap_vizs = list(heatmap_vizs)

    return flows, flow_vizs, heatmaps, heatmap_vizs


def segment_image(
    image: Image,
    flow: np.ndarray,
    sam2_predictor,
    iou_treshold: int = 0.75,
    size_treshold: int = 1000,
):
    normed_flow = np.linalg.norm(flow, axis=2)
    tresh_flow = normed_flow > 1

    filtered_sampling_points = pod_u.get_fitted_sampling_points(
        tresh_flow, points_per_side=32
    )
    # sam2_mask_generator = SAM2AutomaticMaskGenerator(sam2_predictor, pred_iou_thresh=0.7, stability_score_thresh=0.9, point_grids=[filtered_sampling_points], points_per_side=None)
    # sam2_mask_generator = SAM2AutomaticMaskGenerator(sam2_predictor, pred_iou_thresh=0.85, stability_score_thresh=0.9, point_grids=[filtered_sampling_points], points_per_side=None, min_mask_region_area=100)
    sam2_mask_generator = SAM2AutomaticMaskGenerator(
        sam2_predictor,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.6,
        stability_score_offset=2,
        box_nms_thresh=0.3,
        point_grids=[filtered_sampling_points],
        points_per_side=None,
        min_mask_region_area=500,
    )

    masks = sam2_mask_generator.generate(np.array(image.convert("RGB")))

    segmentation_masks = [m["segmentation"] for m in masks]
    segmentation_masks = pod_u.post_process_masks(
        segmentation_masks, iou_treshold=iou_treshold, size_treshold=size_treshold
    )

    # intersect flow and seg
    # Don't need to filter strictly with iou as all segmentation masks originated from sample points within optical flow treshold
    moving_masks, bounding_boxes = pod_u.intersect_flow_seg(
        segmentation_masks,
        tresh_flow,
        bounding_box_bloat=2,
        intersection_treshold=0.35,
    )

    old_boxes = bounding_boxes.copy()

    boxes, mask_centers = pod_u.reorder_masks_compute_mask_centers(
        moving_masks, bounding_boxes
    )
    return moving_masks, boxes, mask_centers, old_boxes


def propagate_mask_through_video(
    image_index: int, boxes, sam2_predictor, inference_state
):
    # TODO: Is it okey to comment this here? Don't think so
    sam2_predictor.reset_state(inference_state)

    ann_obj_id = list(range(1, len(boxes) + 1))
    # colors = ["orange", "green", "red", "blue"]

    _, out_obj_ids, out_mask_logits = zip(
        *[
            sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=image_index,
                obj_id=ann_obj_id[i],
                # points=points,
                # labels=labels,
                box=boxes[i],
            )
            for i in range(len(boxes))
        ]
    )

    # call sam2 for mask propagating through whole video snippet
    video_segments = {}  # video_segments contains the per-frame segmentation results
    # backward
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in sam2_predictor.propagate_in_video(
        inference_state, start_frame_idx=image_index, reverse=True
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    # forward
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in sam2_predictor.propagate_in_video(
        inference_state, start_frame_idx=image_index, reverse=False
    ):  # not reverse means forward
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments


def propagate_all_once(
    image_indices: list[int], boxes: list, sam2_predictor, inference_state
):
    sam2_predictor.reset_state(inference_state)

    ann_obj_id = list(range(1, len(boxes) + 1))
    # colors = ["orange", "green", "red", "blue"]

    _, out_obj_ids, out_mask_logits = zip(
        *[
            sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=image_indices[i],
                obj_id=ann_obj_id[i],
                # points=points,
                # labels=labels,
                box=boxes[i],
            )
            for i in range(len(boxes))
        ]
    )

    # call sam2 for mask propagating through whole video snippet
    video_segments = {}  # video_segments contains the per-frame segmentation results
    # backward
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in sam2_predictor.propagate_in_video(
        inference_state, start_frame_idx=np.max(image_indices), reverse=True
    ):  # start reverse tracking from the last annotated image
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    # forward
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in sam2_predictor.propagate_in_video(
        inference_state, start_frame_idx=np.min(image_indices), reverse=False
    ):  # start forward tracking from the first annotated image
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments


def add_masklets_and_pointers_to_dict(
    video_segments: dict,
    inference_state: dict,
    object_dict: dict,
    pointers_dict: dict,
    num_found_objects: int,
):
    object_id_mapping = {
        i: i + num_found_objects for i in range(1, len(video_segments[0].keys()) + 1)
    }

    for frame_number, frame_masklets in video_segments.items():
        frame_outputs_key = (
            "non_cond_frame_outputs"
            if frame_number
            in inference_state["output_dict"]["non_cond_frame_outputs"].keys()
            else "cond_frame_outputs"
        )
        object_pointers = inference_state["output_dict"][frame_outputs_key][
            frame_number
        ]["obj_ptr"]
        for i, (obj_id, obj_mask) in enumerate(frame_masklets.items()):
            object_dict[frame_number][object_id_mapping[obj_id]] = obj_mask
            pointers_dict[frame_number][object_id_mapping[obj_id]] = object_pointers[i]

    num_found_objects += len(video_segments[0].keys())
    return num_found_objects


def add_all_masklets_and_pointers_to_dict(
    video_segments: dict, inference_state: dict, object_dict: dict, pointers_dict: dict
):
    for frame_number, frame_masklets in video_segments.items():
        frame_outputs_key = (
            "non_cond_frame_outputs"
            if frame_number
            in inference_state["output_dict"]["non_cond_frame_outputs"].keys()
            else "cond_frame_outputs"
        )
        object_pointers = inference_state["output_dict"][frame_outputs_key][
            frame_number
        ]["obj_ptr"].squeeze(0)
        for i in sorted(frame_masklets.keys()):
            object_dict[frame_number][i] = frame_masklets[i]
            try:
                pointers_dict[frame_number][i] = object_pointers[i]
            except:
                pointers_dict[frame_number][i] = np.nan


def convert_dicts_to_arrays(
    object_dict: dict, pointers_dict: dict, num_found_objects: int, img_size: tuple[int]
):
    # Do this at the end once to convert the dict of frame_number, object_id to a list of
    # object_id, frame_number
    H, W = img_size

    object_array = np.empty((num_found_objects, len(object_dict), H, W), dtype=int)
    pointers_array = np.empty((num_found_objects, len(object_dict), 256), dtype=float)

    for frame_number, frame_masklets in sorted(object_dict.items()):
        for obj_id, (obj_mask, obj_ptr) in enumerate(
            zip(frame_masklets.values(), pointers_dict[frame_number].values())
        ):
            # object_array[obj_id - 1][frame_number] = obj_mask
            # pointers_array[obj_id - 1][frame_number] = obj_ptr.cpu().numpy()
            object_array[obj_id][frame_number] = obj_mask
            pointers_array[obj_id][frame_number] = obj_ptr.cpu().numpy()

    return object_array, pointers_array


def convert_dicts_to_arrays_in_order(
    object_dict: dict, pointers_dict: dict, img_size: tuple[int]
):
    # Do this at the end once to convert the dict of frame_number, object_id to a list of
    # object_id, frame_number
    H, W = img_size
    num_images = len(object_dict)
    num_found_objects = len(object_dict[0])

    object_array = np.empty((num_found_objects, num_images, H, W), dtype=int)
    pointers_array = np.empty((num_found_objects, num_images, 256), dtype=float)

    # for frame_number, frame_masklets in sorted(object_dict.items()):
    #     for obj_id, (obj_mask, obj_ptr) in enumerate(zip(frame_masklets.values(), pointers_dict[frame_number].values())):
    #         object_array[obj_id-1][frame_number] = obj_mask
    #         pointers_array[obj_id-1][frame_number] = obj_ptr.cpu().numpy()
    for frame_number in sorted(object_dict.keys()):
        frame_masklets = object_dict[frame_number]
        frame_pointers = pointers_dict[frame_number]
        for i, obj_id in enumerate(sorted(frame_masklets.keys())):
            object_array[i][frame_number] = frame_masklets[obj_id]
            pointers_array[i][frame_number] = frame_pointers[obj_id].cpu().numpy()

    return object_array, pointers_array


def match_masks_against_known_masks(
    new_masks: list, known_masks: dict, iou_threshold: float = 0.5
):
    unseen_masks_indices = list()
    known_mask_keys_to_remove = list()
    for i, new_mask in enumerate(new_masks):
        unseen = True
        for obj_id, known_mask in known_masks.items():
            if (
                np.logical_and(new_mask, known_mask).sum()
                / np.logical_or(new_mask, known_mask).sum()
                > iou_threshold
            ):
                if np.sum(new_mask) / np.sum(known_mask) <= 0.75:
                    # Remove known, bigger mask for smaller, more crisp mask
                    known_mask_keys_to_remove.append(obj_id)
                else:
                    # Else just don't track the newly found one
                    unseen = False
                    break
        if unseen:
            unseen_masks_indices.append(i)

    return unseen_masks_indices, known_mask_keys_to_remove


def find_zero_crossings(signal: np.ndarray):
    """
    Finds the indices where the signal crosses zero.
    """
    sign_changes = np.sign(signal[:-1]) * np.sign(
        signal[1:]
    )  # Check sign change between samples
    zero_crossings = np.where(sign_changes < 0)[0]  # Indices where sign change happens
    return zero_crossings


def get_most_relevant_images(flows, k: int = 10):
    """
    Returns ``k`` image indices of zero crossings of the first derivative with the highest
    absolute values of the second derivative.
    """
    flow_norm_mean = np.mean(
        np.linalg.norm(flows, axis=-1), axis=(1, 2)
    )  # compute mean of flow norm
    flow_norm_mean = gaussian_filter1d(flow_norm_mean, sigma=3)  # smooth flow

    # compute derivatives
    flow_norm_grad = np.gradient(flow_norm_mean)
    flow_norm_grad_grad = np.gradient(flow_norm_grad)

    zero_crossing_grad = find_zero_crossings(flow_norm_grad)

    # get k strongest zero crossings
    best_zero_crossings_grad = np.argsort(
        np.abs(flow_norm_grad_grad[zero_crossing_grad])
    )[-k:]
    best_zero_crossings_grad = zero_crossing_grad[best_zero_crossings_grad]

    return best_zero_crossings_grad


def remove_non_table_objects(
    moving_masks: list, boxes: list, mask_centers: list, old_boxes: list
):
    table_object_indices = [
        i
        for i in range(len(mask_centers))
        if mask_centers[i][1] >= 80 and mask_centers[i][0] <= 412
    ]
    moving_masks = [moving_masks[i] for i in table_object_indices]
    boxes = [boxes[i] for i in table_object_indices]
    mask_centers = [mask_centers[i] for i in table_object_indices]
    old_boxes = [old_boxes[i] for i in table_object_indices]

    return moving_masks, boxes, mask_centers, old_boxes


def main(
    image_filenames: list[str],
    sea_raft_model,
    sea_raft_args,
    sam2_predictor,
    # mask_diff_model_path: str,
    top_k_zero_crossings: int = 10,
):
    # sea_raft_model, sea_raft_args, sam2_predictor, inference_state, mask_diff_model = (
    #     prepare_models_input(image_filenames, mask_diff_model_path)
    # )
    # TODO: Use mask diff model for inferring mask types

    # prepare images and load it into sam2
    tmp_folder = "/app/ncb/tmp_sam2_folder"
    image_filenames = pod_u.clean_and_copy_images(tmp_folder, image_filenames)
    inference_state = sam2_predictor.init_state(
        video_path=tmp_folder, async_loading_frames=False
    )

    flows, flow_vizs, heatmaps, heatmap_vizs = estimate_flow_on_images(
        sea_raft_model, sea_raft_args, image_filenames
    )

    # TODO: Add back in if need to restrain flow
    # crop flow to image segment containing the table
    # for i in range(len(flows)):
    #     flows[i][0:80, :, :] = np.array([0, 0])
    #     flows[i][:, 412:, :] = np.array([0, 0])

    img_size = flows[0].shape[0:2]  # H, W

    # init object_dict for all frames
    object_dict, pointers_dict = dict(), dict()
    num_found_objects = 0
    for i in range(len(image_filenames)):
        object_dict[i] = dict()
        pointers_dict[i] = dict()

    # for image_index in range(len(image_filenames) - 1): # Got one less flow image than actual images. Here, just don't look at the last image
    for image_index in get_most_relevant_images(
        flows, k=top_k_zero_crossings
    ):  # only look at k best images based on flow
        # print(image_index)
        # get moving object masks from image
        image = Image.open(image_filenames[image_index])
        flow = flows[image_index]
        try:
            moving_masks, boxes, mask_centers, old_boxes = segment_image(
                image, flow, sam2_predictor, iou_treshold=0.75, size_treshold=500
            )
        except Exception as e:
            print("Caught an exception during single image segmentation:")
            traceback.print_exc()
            continue

        # remove masks that are not centered on table
        # moving_masks, boxes, mask_centers, old_boxes = remove_non_table_objects(
        #     moving_masks, boxes, mask_centers, old_boxes
        # )

        # discard masks we already saw
        unseen_indices, known_mask_keys_to_remove = match_masks_against_known_masks(
            moving_masks, object_dict[image_index], iou_threshold=0.5
        )
        moving_masks = [moving_masks[i] for i in unseen_indices]
        boxes = [boxes[i] for i in unseen_indices]
        old_boxes = [old_boxes[i] for i in unseen_indices]
        mask_centers = [mask_centers[i] for i in unseen_indices]

        # delete the object in every frame
        # for k in known_mask_keys_to_remove:
        #     for v in object_dict.values():
        #         del v[k]

        # plot all filtered boxes
        # pod_u.plot_image_bounding_box_and_mask(image, moving_masks, old_boxes)

        # if nothing new is found in image, continue
        if len(moving_masks) == 0:
            continue

        # propagate new masks through image
        video_segments = propagate_mask_through_video(
            image_index, boxes, sam2_predictor, inference_state
        )
        num_found_objects = add_masklets_and_pointers_to_dict(
            video_segments,
            inference_state,
            object_dict,
            pointers_dict,
            num_found_objects,
        )

        # pod_u.plot_sam2_masks_propagation(image_filenames, video_segments, vis_frame_stride=25)

    # print("Final masks:")
    # pod_u.plot_sam2_masks_propagation(image_filenames, object_dict, vis_frame_stride=25)

    # Do this at the end once to convert the dict of frame_number, object_id to a list
    # of object_id, frame_number
    object_array, pointers_array = convert_dicts_to_arrays(
        object_dict, pointers_dict, num_found_objects, img_size
    )
    # print(object_array.shape)

    return object_array, pointers_array, inference_state, image_filenames


def run_propagation_once_main(
    image_filenames: list[str],
    mask_diff_model_path: str,
    top_k_zero_crossings: int = 10,
):
    sea_raft_model, sea_raft_args, sam2_predictor, inference_state, mask_diff_model = (
        prepare_models(image_filenames, mask_diff_model_path)
    )

    flows, flow_vizs, heatmaps, heatmap_vizs = estimate_flow_on_images(
        sea_raft_model, sea_raft_args, image_filenames
    )

    # crop flow to image segment containing the table
    for i in range(len(flows)):
        flows[i][0:80, :, :] = np.array([0, 0])
        flows[i][:, 412:, :] = np.array([0, 0])

    img_size = flows[0].shape[0:2]  # H, W

    # init object_dict for all frames
    object_dict, pointers_dict = dict(), dict()
    for i in range(len(image_filenames)):
        object_dict[i] = dict()
        pointers_dict[i] = dict()

    tracking_boxes = list()
    tracking_image_indices = list()
    for image_index in get_most_relevant_images(
        flows, k=top_k_zero_crossings
    ):  # only look at k best images based on flow
        print(image_index)
        # get moving object masks from image
        image = Image.open(image_filenames[image_index])
        flow = flows[image_index]
        try:
            moving_masks, boxes, mask_centers, old_boxes = segment_image(
                image, flow, sam2_predictor, iou_treshold=0.75, size_treshold=1000
            )
        except Exception as e:
            print("Caught an exception during single image segmentation:")
            traceback.print_exc()
            continue

        # TODO: Add this back if need to ignore iamge segments
        # remove masks that are not centered on table
        # moving_masks, boxes, mask_centers, old_boxes = remove_non_table_objects(
        #     moving_masks, boxes, mask_centers, old_boxes
        # )

        # plot all filtered boxes
        pod_u.plot_image_bounding_box_and_mask(image, moving_masks, old_boxes)

        # if nothing new is found in image, continue
        if len(moving_masks) == 0:
            continue
        else:
            tracking_boxes += boxes  # add list of found boxes to list of boxes that should be tracked by SAM2 later
            tracking_image_indices += [image_index] * len(
                boxes
            )  # add image index of found objects

    print(tracking_image_indices)
    print(tracking_boxes)

    # propagate all found masks through video at once
    video_segments = propagate_all_once(
        tracking_image_indices, tracking_boxes, sam2_predictor, inference_state
    )
    add_all_masklets_and_pointers_to_dict(
        video_segments, inference_state, object_dict, pointers_dict
    )

    # pod_u.plot_sam2_masks_propagation(image_filenames, video_segments, vis_frame_stride=25)

    pod_u.plot_sam2_masks_propagation(image_filenames, object_dict, vis_frame_stride=25)

    # Do this at the end once to convert the dict of frame_number, object_id to a list
    # of object_id, frame_number
    object_array, pointers_array = convert_dicts_to_arrays_in_order(
        object_dict, pointers_dict, img_size
    )
    print(object_array.shape)

    return object_array, pointers_array, inference_state


if __name__ == "__main__":
    main()
