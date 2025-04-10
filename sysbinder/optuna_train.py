import os
import math
import argparse
import numpy as np
import torch
import torchvision.utils as vutils
import json
from pytorch_metric_learning.losses import NTXentLoss
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel as DP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import optuna


from sysbinder.sysbinder import SysBinderImageAutoEncoder
from sysbinder.data import GlobDataset, ContrastiveBatchSampler
from sysbinder.utils_sysbinder import linear_warmup, cosine_anneal, set_seed


torch.set_num_threads(10)
OMP_NUM_THREADS = 10
MKL_NUM_THREADS = 10


parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=40)
parser.add_argument("--num_workers", type=int, default=4)
# parser.add_argument('--image_size', type=int, default=128)
parser.add_argument("--image_height", type=int, default=160)
parser.add_argument("--image_width", type=int, default=240)
parser.add_argument("--image_channels", type=int, default=3)

parser.add_argument("--checkpoint_path", default="checkpoint.pt.tar")
parser.add_argument("--data_path", default="data/*.png")
parser.add_argument("--log_path", default="logs/")

parser.add_argument("--lr_dvae", type=float, default=3e-4)
parser.add_argument("--lr_enc", type=float, default=1e-4)
parser.add_argument("--lr_dec", type=float, default=3e-4)
parser.add_argument("--lr_warmup_steps", type=int, default=30000)
parser.add_argument("--lr_half_life", type=int, default=250000)
parser.add_argument("--clip", type=float, default=0.05)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--ce_weight", type=float, default=1.0)

parser.add_argument("--num_iterations", type=int, default=3)
parser.add_argument("--num_slots", type=int, default=4)
parser.add_argument("--num_blocks", type=int, default=8)
parser.add_argument("--cnn_hidden_size", type=int, default=512)
parser.add_argument("--slot_size", type=int, default=2048)
parser.add_argument("--mlp_hidden_size", type=int, default=192)
parser.add_argument("--num_prototypes", type=int, default=64)
parser.add_argument(
    "--temp", type=float, default=1.0, help="softmax temperature for prototype binding"
)
parser.add_argument("--temp_step", default=False, action="store_true")

parser.add_argument("--vocab_size", type=int, default=4096)
parser.add_argument("--num_decoder_layers", type=int, default=8)
parser.add_argument("--num_decoder_heads", type=int, default=4)
parser.add_argument("--d_model", type=int, default=192)
parser.add_argument("--dropout", type=float, default=0.1)

parser.add_argument("--tau_start", type=float, default=1.0)
parser.add_argument("--tau_final", type=float, default=0.1)
parser.add_argument("--tau_steps", type=int, default=30000)

parser.add_argument("--use_dp", default=False, action="store_true")

parser.add_argument(
    "--binarize",
    default=False,
    action="store_true",
    help="Should the encodings of the sysbinder be binarized?",
)
parser.add_argument(
    "--attention_codes",
    default=False,
    action="store_true",
    help="Should the sysbinder prototype attention values be used as encodings?",
)
parser.add_argument(
    "--contrastive_loss",
    default=False,
    action="store_true",
    help="Enable contrastive loss",
)
parser.add_argument(
    "--contrastive_loss_temp",
    type=float,
    default=0.4,
    help="Temperature of the contrastive loss NT-Xent",
)
parser.add_argument(
    "--contrastive_loss_weight",
    type=float,
    default=0.1,
    help="Weight of the contrastive loss contributing to the overall loss",
)


def get_args():
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    return args


def train_and_evaluate(trial, args):

    # Optuna suggests hyperparameters
    args.lr_dvae = trial.suggest_float("lr_dvae", 1e-9, 1e-5, log=True)
    args.lr_enc = trial.suggest_float("lr_enc", 1e-5, 1e-1, log=True)
    args.lr_dec = trial.suggest_float("lr_dec", 1e-9, 1e-5, log=True)
    args.tau_start = trial.suggest_float("tau_start", 0.5, 1.1)
    args.tau_final = trial.suggest_float("tau_final", 0.1, 0.5)
    args.tau_steps = trial.suggest_int("tau_steps", 1, 10)
    # args.ce_weight = trial.suggest_float("ce_weight", 1, 4)

    set_seed(args.seed)

    # a single video has 200k images
    train_dataset = GlobDataset(
        root=args.data_path,
        phase="train",
        # phase=15000,    # get 15000 random samples instead for UT Ego
        img_height=args.image_height,
        img_width=args.image_width,
    )
    val_dataset = GlobDataset(
        root=args.data_path,
        phase="val",
        # phase = 5000,   # get 5000 random samples instead for UT Ego
        img_height=args.image_height,
        img_width=args.image_width,
    )

    train_sampler = None
    val_sampler = None
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }

    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

    train_epoch_size = int(np.ceil(len(train_dataset) / args.batch_size))
    val_epoch_size = int(np.ceil(len(val_dataset) / args.batch_size))

    model = SysBinderImageAutoEncoder(args)

    if os.path.isfile(args.checkpoint_path):
        print(f"Loading checkpoint {args.checkpoint_path}...")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        try:
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint["best_val_loss"]
            best_epoch = checkpoint["best_epoch"]
            model.load_state_dict(checkpoint["model"])
            model.image_encoder.sysbinder.prototype_memory.attn.temp = checkpoint[
                "temp"
            ]
        # when the checkpoint is the best model it contains only the model state
        except:
            try:
                model.load_state_dict(checkpoint)
                if args.temp_step:
                    model.image_encoder.sysbinder.prototype_memory.attn.temp = 0.001
                elif args.temp_step == False and args.temp != 1.0:
                    model.image_encoder.sysbinder.prototype_memory.attn.temp = 1e-4
                else:
                    model.image_encoder.sysbinder.prototype_memory.attn.temp = 1.0
            # unless a later version was used
            except:
                model.load_state_dict(checkpoint["model"])
                model.image_encoder.sysbinder.prototype_memory.attn.temp = checkpoint[
                    "temp"
                ]
            start_epoch = 0
            best_val_loss = math.inf
            best_epoch = 0
        print(f"Checkpoint {args.checkpoint_path} loaded")
    else:
        checkpoint = None
        start_epoch = 0
        best_val_loss = math.inf
        best_epoch = 0

    model = model.to(args.device)
    if args.use_dp:
        model = DP(model)

    # TODO: What does this do?
    optimizer = Adam(
        [
            {
                "params": (x[1] for x in model.named_parameters() if "dvae" in x[0]),
                "lr": args.lr_dvae,
            },
            {
                "params": (
                    x[1] for x in model.named_parameters() if "image_encoder" in x[0]
                ),
                "lr": 0.0,
            },
            {
                "params": (
                    x[1] for x in model.named_parameters() if "image_decoder" in x[0]
                ),
                "lr": 0.0,
            },
        ]
    )

    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        # when the checkpoint is the best model it contains only the model state
        except:
            pass

    first_val_loss = None

    for epoch in range(start_epoch, args.epochs):
        model.train()

        # train step
        for batch, (image, filenames) in enumerate(
            train_loader
        ):  # filenames are a tuple of image filenames
            global_step = epoch * train_epoch_size + batch

            tau = cosine_anneal(
                global_step, args.tau_start, args.tau_final, 0, args.tau_steps
            )

            lr_warmup_factor_enc = linear_warmup(
                global_step, 0.0, 1.0, 0.0, args.lr_warmup_steps
            )

            lr_warmup_factor_dec = linear_warmup(
                global_step, 0.0, 1.0, 0, args.lr_warmup_steps
            )

            lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

            optimizer.param_groups[0]["lr"] = args.lr_dvae
            optimizer.param_groups[1]["lr"] = (
                lr_decay_factor * lr_warmup_factor_enc * args.lr_enc
            )
            optimizer.param_groups[2]["lr"] = (
                lr_decay_factor * lr_warmup_factor_dec * args.lr_dec
            )

            image = image.to(args.device)

            optimizer.zero_grad()

            (recon_dvae, cross_entropy, mse, attns, emb_set) = model(image, tau)

            if args.use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()

            loss = mse + args.ce_weight * cross_entropy

            loss.backward()

            clip_grad_norm_(model.parameters(), args.clip, "inf")

            optimizer.step()

        with torch.no_grad():
            model.eval()

            val_cross_entropy = 0.0
            val_mse = 0.0

            # validation step
            for batch, (image, filenames) in enumerate(val_loader):
                image = image.to(args.device)

                (recon_dvae, cross_entropy, mse, attns, emb_set) = model(image, tau)

                if args.use_dp:
                    mse = mse.mean()
                    cross_entropy = cross_entropy.mean()

                val_cross_entropy += cross_entropy.item()
                val_mse += mse.item()

            val_cross_entropy /= val_epoch_size
            val_mse /= val_epoch_size

            val_loss = val_mse + args.ce_weight * val_cross_entropy

            # Save first val loss to look for convergence
            if first_val_loss is None:
                first_val_loss = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

            # TODO: what does report do? Change its loss to fit too
            trial.report(val_loss/first_val_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            torch.cuda.empty_cache()
    
    print(f"First val loss: {first_val_loss}, Last val loss: {val_loss}, Best val loss: {best_val_loss}")
    return val_loss, first_val_loss


def objective(trial):
    args = get_args()
    args.epochs = 10    # 10
    val_loss, first_val_loss = train_and_evaluate(trial, args)
    # return val_loss    # for normal hp
    
    # make sure that training converges lol
    return val_loss / first_val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, timeout=None, show_progress_bar=True)  # Run 12 trials
    print("Best Hyperparameters:", study.best_params)
