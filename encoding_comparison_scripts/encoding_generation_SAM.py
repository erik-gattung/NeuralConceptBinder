# handle imports, init ncb
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import utils_ncb as utils_bnr
from argparse import Namespace
from pathlib import Path

from sam_concept_binder import SAMConceptBinder
import sam2_pipeline_main


def get_ncb_encoding(sam_ncb_model, img_fps, sea_raft_model, sea_raft_args, sam2_predictor):
    # codes, probs, object_array, inference_state = ncb_model.encode(img_fps)
    # return codes, probs, object_array, inference_state
    object_array, pointers_array, inference_state, image_filenames, hard_codes = sam_ncb_model.encode(
        img_fps, sea_raft_model, sea_raft_args, sam2_predictor
    )
    return object_array, pointers_array, inference_state, image_filenames, hard_codes


def main(video_range: range):
    # prepare ncb retrieval corpus
    ncb_seed = 1

    # setup ncb
    utils_bnr.set_seed(0)
    args = Namespace(
        seed=0,
        batch_size=20,
        num_workers=0,
        # image_size=128,
        image_height=128,
        image_width=128,
        image_channels=3,
        data_path="data/*.png",
        perc_imgs=1.0,
        log_path="../logs/",
        # checkpoint_path=f"/app/ncb/CLEVR-4/retbind_seed_{ncb_seed}/best_model.pt",
        # checkpoint_path=f"/app/ncb/CLEVR-4/CLEVRER_retbind_seed_{ncb_seed}/best_model.pt",
        checkpoint_path=f"/app/ncb/CLEVR-4/SAM2_retbind_seed_{ncb_seed}/best_model.pt",
        # checkpoint_path=f"/app/ncb/logs/clevr-4_rect/best_model.pt",
        # checkpoint_path=f"/app/ncb/logs/clevrer_seed_{ncb_seed}/best_model.pt",
        # checkpoint_path=f"/app/ncb/logs/car_parts_seed_{ncb_seed}/best_model.pt",
        model_type="ncb",
        use_dp=False,
        name="ncb_encoding_eval",
        num_categories=3,
        clf_type="dt",
        lr_dvae=0.0003,
        lr_enc=0.0001,
        lr_dec=0.0003,
        lr_warmup_steps=30000,
        lr_half_life=250000,
        clip=0.05,
        epochs=500,
        num_iterations=3,
        num_slots=4,
        num_blocks=16,
        cnn_hidden_size=512,
        slot_size=2048,
        mlp_hidden_size=192,
        num_prototypes=64,
        temp=1.0,
        temp_step=False,
        vocab_size=4096,
        num_decoder_layers=8,
        num_decoder_heads=4,
        d_model=192,
        dropout=0.1,
        tau_start=1.0,
        tau_final=0.1,
        tau_steps=30000,
        lr=0.01,
        binarize=False,
        attention_codes=False,
        # retrieval_corpus_path=f"/app/ncb/CLEVR-4/retbind_seed_{ncb_seed}/block_concept_dicts.pkl",
        # retrieval_corpus_path=f"/app/ncb/CLEVR-4/CLEVRER_retbind_seed_{ncb_seed}/block_concept_dicts.pkl",
        retrieval_corpus_path=f"/app/ncb/CLEVR-4/SAM2_retbind_seed_{ncb_seed}/block_concept_dicts.pkl",
        # retrieval_corpus_path="/app/ncb/logs/clevr-4_rect/block_concept_dicts.pkl",
        # retrieval_corpus_path=f"/app/ncb/logs/clevrer_seed_{ncb_seed}/block_concept_dicts.pkl",
        # retrieval_corpus_path=f"/app/ncb/logs/car_parts_seed_{ncb_seed}/block_concept_dicts.pkl",
        deletion_dict_path=None,
        merge_dict_path=None,
        retrieval_encs="proto-exem",
        majority_vote=False,
        topk=5,
        thresh_attn_obj_slots=0.98,
        thresh_count_obj_slots=4,  # -1 for all
        device="cuda",
        SAM_slot_size = 256,
        SAM_num_blocks = 8,
    )
    if args.model_type == "ncb":
        ncb_model = SAMConceptBinder(args)
    else:
        raise ValueError(f"Model type {args.model_type} not handled in this script!")
    ncb_model.to(args.device)
    ncb_model.eval()
    torch.set_grad_enabled(True)

    sea_raft_model, sea_raft_args, sam2_predictor = (
        sam2_pipeline_main.prepare_models()
    )

    # init encoding dict
    root = Path("/app/clevrer_videos/video_frames_val/")
    hard_encoding_dict = dict()
    soft_encoding_dict = dict()

    for video_number in video_range:
        base_path = os.path.join(root, f"video_{video_number}")
        frame_names = sorted([fn for fn in os.listdir(base_path) if ".jpg" in fn])
        frame_names = [os.path.join(base_path, frame_name) for frame_name in frame_names]

        hard_encoding_dict[video_number] = dict()
        soft_encoding_dict[video_number] = dict()

        object_array, pointers_array, inference_state, image_filenames, hard_codes = get_ncb_encoding(ncb_model, frame_names, sea_raft_model, sea_raft_args, sam2_predictor)

        for image_number in range(len(image_filenames)):
            hard_encoding_dict[video_number][str(image_number)] = hard_codes[:, image_number, :].cpu().numpy().astype(int)
            soft_encoding_dict[video_number][str(image_number)] = pointers_array[:, image_number, :]
            

    # save encodings as pickle
    with open(f"/app/ncb/encoding_comparison_scripts/results/SAM/val_hard_encodings_{video_range[0]}-{video_range[-1]}.pkl", "wb") as f:
        pickle.dump(hard_encoding_dict, f)

    with open(f"/app/ncb/encoding_comparison_scripts/results/SAM/val_soft_encodings_{video_range[0]}-{video_range[-1]}.pkl", "wb") as f:
        pickle.dump(soft_encoding_dict, f)


if __name__ == "__main__":
    video_range = range(10000, 10001)
    main(video_range)
