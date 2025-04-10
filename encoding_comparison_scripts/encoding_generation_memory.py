# handle imports, init ncb
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
import os
import utils_ncb as utils_bnr
from torchvision import transforms
from argparse import Namespace
from sysbinder.sysbinder import SysBinderImageAutoEncoder
from pathlib import Path
from collections import defaultdict

# from neural_concept_binder import NeuralConceptBinder
from memory_concept_binder import MemoryConceptBinder


def get_ncb_encoding(
    model: SysBinderImageAutoEncoder, args: Namespace, img_fn: str):
    # TODO: Why do I need to set a seed here for consistent results? I think because of the random slot initialization
    utils_bnr.set_seed(0)

    img = Image.open(img_fn).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
        ]
    )
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # add batch dimension

    codes, probs, slots, attns_vis, attns, reset_memory_bank = model.encode(img_tensor)

    return codes, probs, slots, attns_vis, attns, reset_memory_bank

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
        checkpoint_path=f"/app/ncb/CLEVR-4/retbind_seed_{ncb_seed}/best_model.pt",
        # checkpoint_path=f"/app/ncb/CLEVR-4/CLEVRER_retbind_seed_{ncb_seed}/best_model.pt",
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
        retrieval_corpus_path=f"/app/ncb/CLEVR-4/retbind_seed_{ncb_seed}/block_concept_dicts.pkl",
        # retrieval_corpus_path=f"/app/ncb/CLEVR-4/CLEVRER_retbind_seed_{ncb_seed}/block_concept_dicts.pkl",
        deletion_dict_path=None,
        merge_dict_path=None,
        retrieval_encs="proto-exem",
        majority_vote=False,
        topk=5,
        thresh_attn_obj_slots=0.98,
        thresh_count_obj_slots=-1,  # -1 for all TODO: What if I change this to only the relevant ones?
        device="cuda",
    )
    utils_bnr.set_seed(0)
    if args.model_type == "ncb":
        ncb_model = MemoryConceptBinder(args)
    else:
        raise ValueError(f"Model type {args.model_type} not handled in this script!")
    ncb_model.to(args.device)
    ncb_model.eval()
    torch.set_grad_enabled(True)

    # init encoding dict
    # root = Path("/app/clevrer_videos/square_frames_val")
    # root = Path("/app/custom_clevrer/square_images")
    root = Path("/app/custom_clevrer/square_face_images")
    hard_encoding_dict = dict()
    soft_encoding_dict = dict()
    reset_memory_bank_dict = dict()


    all_image_files = [Path.joinpath(root, f.name) for f in root.iterdir() if f.suffix.lower()[1:] in ["jpg", "jpeg", "png"]]

    video_dict = defaultdict(list)

    for image in all_image_files:
        try:
            _, vid_num, _, img_num = image.stem.split("_")  # `stem` removes the suffix
            video_dict[int(vid_num)].append(image)
        except ValueError:
            continue  # Skip files that don't match the expected pattern

    # Sort each list and store in final output
    sorted_image_files = [sorted(video_dict[v]) for v in video_range if v in video_dict]

    print(f"Number of admissible videos found: {len(sorted_image_files)}")

    for image_list in sorted_image_files:
        ncb_model.reset_memory()    # wipe the memory for each new video
        _, video_number, _, _ = image.stem.split("_")
        if video_number not in hard_encoding_dict.keys():
            hard_encoding_dict[video_number] = dict()
            soft_encoding_dict[video_number] = dict()
            reset_memory_bank_dict[video_number] = dict()

        for image in image_list:
            _, _, _, image_number = image.stem.split("_")

            codes, _, cont_codes, _, _, reset_memory_bank = get_ncb_encoding(ncb_model, args, image)
            hard_encoding_dict[video_number][image_number] = codes.cpu().squeeze().numpy().astype(int)
            soft_encoding_dict[video_number][image_number] = cont_codes.cpu().detach().squeeze().numpy()
            reset_memory_bank_dict[video_number][image_number] = reset_memory_bank

    # save encodings as pickle
    with open(f"/app/ncb/encoding_comparison_scripts/results/memory_v3_custom_complex_CLEVR/val_hard_encodings_{video_range[0]}-{video_range[-1]}.pkl", "wb") as f:
        pickle.dump(hard_encoding_dict, f)

    with open(f"/app/ncb/encoding_comparison_scripts/results/memory_v3_custom_complex_CLEVR/val_soft_encodings_{video_range[0]}-{video_range[-1]}.pkl", "wb") as f:
        pickle.dump(soft_encoding_dict, f)

    with open(f"/app/ncb/encoding_comparison_scripts/results/memory_v3_custom_complex_CLEVR/reset_memory_bank_{video_range[0]}-{video_range[-1]}.pkl", "wb") as f:
        pickle.dump(reset_memory_bank_dict, f)


if __name__ == "__main__":
    # video_range = range(10000, 10010)
    video_range = range(0, 8)
    main(video_range)
