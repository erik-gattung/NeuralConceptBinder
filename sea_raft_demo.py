import argparse
import cv2
import numpy as np

import torch
import torch.nn.functional as F

from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt


import json
import argparse


def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, "r") as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args


def parse_args(parser, args_list=None):
    if args_list is not None:
        entry = parser.parse_args(args_list)
    else:
        entry = parser.parse_args()
    json_path = entry.cfg
    args = json_to_args(json_path)
    args_dict = args.__dict__
    for index, (key, value) in enumerate(vars(entry).items()):
        args_dict[key] = value
    return args


def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar


def add_color_bar_to_image(image, color_bar, orientation="vertical"):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == "vertical":
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])


def vis_heatmap(image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(
        50, width, cv2.COLORMAP_JET
    )  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, "vertical")
    return combined_image


def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap


def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output["flow"][-1]
    info_final = output["info"][-1]
    return flow_final, info_final


def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(
        image1, scale_factor=2**args.scale, mode="bilinear", align_corners=False
    )
    img2 = F.interpolate(
        image2, scale_factor=2**args.scale, mode="bilinear", align_corners=False
    )
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(
        flow, scale_factor=0.5**args.scale, mode="bilinear", align_corners=False
    ) * (0.5**args.scale)
    info_down = F.interpolate(info, scale_factor=0.5**args.scale, mode="area")
    return flow_down, info_down


@torch.no_grad()
def estimate_flow(model, args, image1: str, image2: str, device="cuda"):
    image1 = cv2.imread(image1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(image2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)

    flow, info = calc_flow(args, model, image1, image2)
    flow_vis = flow_to_image(
        flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False
    )
    heatmap = get_heatmap(info, args)
    heatmap_vis = vis_heatmap(
        image1[0].permute(1, 2, 0).cpu().numpy(),
        heatmap[0].permute(1, 2, 0).cpu().numpy(),
    )
    return flow, flow_vis, heatmap, heatmap_vis


def create_model(args):
    if args.path is None and args.url is None:
        raise ValueError("Either --path or --url must be provided")
    if args.path is not None:
        model = RAFT(args)
        load_ckpt(model, args.path)
    else:
        model = RAFT.from_pretrained(args.url, args=args)

    if args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    return model
