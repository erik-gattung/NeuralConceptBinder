import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sysbinder.sysbinder import SysBinderImageAutoEncoder

from data import CLEVREasy_1_WithAnnotations, CLEVR4_1_WithAnnotations
from neural_concept_binder import NeuralConceptBinder

# Baseline, Repository needs to be cloned from https://github.com/yfw/nlotm
import utils_ncb as utils_bnr

torch.set_num_threads(40)
OMP_NUM_THREADS = 40
MKL_NUM_THREADS = 40

SEED = 0

def gather_encs(model, loader, args):
    model.eval()
    torch.set_grad_enabled(True)

    all_labels_multi = []
    all_labels_single = []
    all_codes = []
    for i, sample in tqdm(enumerate(loader)):
        img_locs = sample[-1]
        sample = sample[:-1]
        imgs, _, _, _ = map(
            lambda x: x.to(args.device), sample
        )

        # encode image with whatever model is being used
        codes, probs = model.encode(imgs)

        # make sure only one object/slot per image
        assert codes.shape[0] == args.batch_size
        # codes = codes.squeeze(dim=1)
        all_codes.append(codes.detach().cpu().numpy())

    all_codes = np.concatenate(all_codes, axis=0)

    return all_codes


def get_args():
    args = utils_bnr.get_parser(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ).parse_args()

    utils_bnr.set_seed(SEED)

    return args


def main():
    args = get_args()
    args.retrieval_corpus_path = "/app/ncb/CLEVR-4/retbind_seed_0/block_concept_dicts.pkl"
    args.checkpoint_path = "/app/ncb/CLEVR-4/retbind_seed_0/best_model.pt"
    args.num_blocks = 16
    args.batch_size = 20
    args.clf_type = "dt"
    args.thresh_count_obj_slots = 0 # -1 for all

    image_filename = "/app/ncb/filtered_video_10000/00000.jpg"

    print(f"Delivered args: {args}")
    print(
        f"{args.checkpoint_path} loading for {args.model_type} encoding classification"
    )

    if args.model_type == "ncb":
        model = NeuralConceptBinder(args)
    else:
        raise ValueError(f"Model type {args.model_type} not handled in this script!")

    model.to(args.device)

    print(f"Loading image {image_filename}")
    img = Image.open(image_filename)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)    # add batch dimension

    codes, probs = model.encode(img_tensor)

    print("Gathered encodings of provided data ...")
    print("-------------------------------------------\n")
    print(f"Codes: {codes}")
    print("-------------------------------------------\n")
    print(f"Probs: {probs}")


if __name__ == "__main__":
    main()
