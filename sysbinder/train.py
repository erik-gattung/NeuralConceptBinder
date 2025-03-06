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
from rtpt import RTPT


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


def visualize(image, recon_dvae, recon_tf, attns, N=8):

    # tile
    tiles = torch.cat(
        (
            image[:N, None, :, :, :],
            recon_dvae[:N, None, :, :, :],
            recon_tf[:N, None, :, :, :],
            attns[:N, :, :, :, :],
        ),
        dim=1,
    ).flatten(end_dim=1)

    # grid
    grid = vutils.make_grid(tiles, nrow=(1 + 1 + 1 + args.num_slots), pad_value=0.8)

    return grid


def main(args):
    arg_str_list = ["{}={}".format(k, v) for k, v in vars(args).items()]
    arg_str = "__".join(arg_str_list)
    if args.contrastive_loss:
        log_dir = os.path.join(args.log_path, f"finetuning_{args.contrastive_loss_temp}_{args.contrastive_loss_weight}")
    else:
        log_dir = os.path.join(args.log_path, datetime.today().isoformat())
    writer = SummaryWriter(log_dir)
    writer.add_text("hparams", arg_str)
    with open(os.path.join(writer.log_dir, "commandline_args.txt"), "w") as f:
        for arg in vars(args):
            f.write(f"\n{arg}: {getattr(args, arg)}")

    # train_dataset = GlobDataset(root=args.data_path, phase='train', img_size=args.image_size)
    # val_dataset = GlobDataset(root=args.data_path, phase='val', img_size=args.image_size)
    train_dataset = GlobDataset(
        root=args.data_path,
        phase="train",
        img_height=args.image_height,
        img_width=args.image_width,
    )
    val_dataset = GlobDataset(
        root=args.data_path,
        phase="val",
        img_height=args.image_height,
        img_width=args.image_width,
    )

    if args.contrastive_loss:
        train_sampler = ContrastiveBatchSampler(train_dataset, args.batch_size)
        val_sampler = ContrastiveBatchSampler(val_dataset, args.batch_size)
        loader_kwargs = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "drop_last": True,
        }
    else:
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

    # train_epoch_size = len(train_loader)
    # val_epoch_size = len(val_loader)
    train_epoch_size = int(np.ceil(len(train_dataset) / args.batch_size))
    val_epoch_size = int(np.ceil(len(val_dataset) / args.batch_size))

    log_interval = max(train_epoch_size // 5, 1)

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

    print(
        f"Training for {args.epochs} - {start_epoch} = {args.epochs - start_epoch} epochs ..."
    )

    model = model.to(args.device)
    if args.use_dp:
        model = DP(model)

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

    # Create and start RTPT object
    rtpt = RTPT(
        name_initials="YOURINITIALS",
        experiment_name=f"SysBinderRetriever",
        max_iterations=args.epochs - start_epoch,
    )
    rtpt.start()

    if args.contrastive_loss:
        ntxent_loss = NTXentLoss(temperature=args.contrastive_loss_temp)

    for epoch in range(start_epoch, args.epochs):
        model.train()

        # TODO: comment this in if necessary
        # if block attention temperature scheduler, then reduce by 0.5
        if args.temp_step:
            if epoch > 0 and epoch % 50 == 0:
                model.module.image_encoder.sysbinder.prototype_memory.attn.temp *= 0.5
            print(
                f"Current temperature: {model.module.image_encoder.sysbinder.prototype_memory.attn.temp}"
            )

        # init emb_set_dict to look up emb_set values of images later for the contrastive loss
        # emb_set_dict = dict()
        # TODO: check filenames for labels, move constrative loss to main loss computation, remove rest

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

            # save emb_set from sysbinder for contrastive loss computation
            if args.contrastive_loss:
                video_ids = [fn.split("_")[1] for fn in filenames]
                label_map = {vid: i for i, vid in enumerate(np.unique(video_ids))}
                labels = torch.tensor([label_map[vid] for vid in video_ids]).to(
                    args.device
                )
                embeddings = torch.flatten(emb_set, start_dim=1)

                xent_loss = ntxent_loss(embeddings, labels)

                # args.contrastive_loss_weight *

                # emb_set = emb_set.to("cpu", non_blocking=True)

                # for i, filename in enumerate(filenames):
                #     filename = filename.split("/")[-1]
                #     _, video_id, _, frame_id = filename.split("_")
                #     frame_id = frame_id.split(".")[0]
                #     video_id, frame_id = int(video_id), int(frame_id)

                #     if video_id not in emb_set_dict.keys():
                #         emb_set_dict[video_id] = dict()

                #     emb_set_dict[video_id][frame_id] = emb_set[i, :, :] # 1, H * W, cnn_hidden_size

            if args.use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()
                if args.contrastive_loss:
                    xent_loss = xent_loss.mean()

            loss = mse + cross_entropy
            if args.contrastive_loss:
                loss += args.contrastive_loss_weight * xent_loss

            loss.backward()

            clip_grad_norm_(model.parameters(), args.clip, "inf")

            optimizer.step()

            with torch.no_grad():
                if batch % log_interval == 0:
                    if args.contrastive_loss:
                        print(
                            "Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F} \t NT-Xent Loss: {:F}".format(
                                epoch + 1,
                                batch,
                                train_epoch_size,
                                loss.item(),
                                mse.item(),
                                xent_loss.item(),
                            )
                        )
                        writer.add_scalar(
                            "TRAIN/nt-xent", xent_loss.item(), global_step
                        )
                    else:
                        print(
                            "Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}".format(
                                epoch + 1,
                                batch,
                                train_epoch_size,
                                loss.item(),
                                mse.item(),
                            )
                        )

                    writer.add_scalar("TRAIN/loss", loss.item(), global_step)
                    writer.add_scalar(
                        "TRAIN/cross_entropy", cross_entropy.item(), global_step
                    )
                    writer.add_scalar("TRAIN/mse", mse.item(), global_step)

                    writer.add_scalar("TRAIN/tau", tau, global_step)
                    writer.add_scalar(
                        "TRAIN/lr_dvae", optimizer.param_groups[0]["lr"], global_step
                    )
                    writer.add_scalar(
                        "TRAIN/lr_enc", optimizer.param_groups[1]["lr"], global_step
                    )
                    writer.add_scalar(
                        "TRAIN/lr_dec", optimizer.param_groups[2]["lr"], global_step
                    )

        # Apply contrastive loss
        # if args.contrastive_loss:
        #     half_batch = args.batch_size // 2
        #     iteration_count = 0

        #     while len(emb_set_dict) > 0:
        #         # get two random unique video ids
        #         vid1, vid2 = np.random.choice(list(emb_set_dict.keys()), 2, replace=False)

        #         # try to get up to 8 embeddings (of a frame) from each video
        #         embedding_keys1 = np.random.choice(list(emb_set_dict[vid1].keys()), min(half_batch, len(emb_set_dict[vid1])), replace=False)
        #         embedding_keys2 = np.random.choice(list(emb_set_dict[vid2].keys()), min(half_batch, len(emb_set_dict[vid2])), replace=False)
        #         emb_sets1 = [emb_set_dict[vid1].pop(key) for key in embedding_keys1]
        #         emb_sets2 = [emb_set_dict[vid2].pop(key) for key in embedding_keys2]
        #         if len(emb_set_dict[vid1]) == 0:
        #             emb_set_dict.pop(vid1)
        #         if len(emb_set_dict[vid2]) == 0:
        #             emb_set_dict.pop(vid2)

        #         labels1 = len(emb_sets1) * [0]
        #         labels2 = len(emb_sets2) * [1]

        #         batch_embeddings = torch.stack(emb_sets1 + emb_sets2, dim=0)
        #         batch_embeddings = torch.flatten(batch_embeddings, start_dim=1).to(args.device)
        #         batch_labels = torch.tensor(labels1 + labels2).to(args.device)

        #         # if there somehow are no labels at all, continue
        #         if batch_labels.size()[0] == 0:
        #             continue

        #         optimizer.zero_grad()
        #         xent_loss = args.contrastive_loss_weight * ntxent_loss(batch_embeddings, batch_labels)
        #         xent_loss.backward()
        #         clip_grad_norm_(model.parameters(), args.clip, 'inf')
        #         optimizer.step()

        # with torch.no_grad():
        #     if iteration_count % log_interval == 0:
        #         print('Train Epoch: {:3} \t NT-Xent Loss: {:F}'.format(
        #                 epoch+1, xent_loss.item()))

        #         writer.add_scalar('TRAIN/nt-xent', xent_loss.item(), global_step)

        # iteration_count += 1

        with torch.no_grad():
            recon_tf = (
                model.module if args.use_dp else model
            ).reconstruct_autoregressive(image[:8])
            grid = visualize(image, recon_dvae, recon_tf, attns, N=8)
            writer.add_image("TRAIN_recons/epoch={:03}".format(epoch + 1), grid)

        with torch.no_grad():
            model.eval()

            val_cross_entropy = 0.0
            val_mse = 0.0
            if args.contrastive_loss:
                val_nt_xent = 0.0

            # init emb_set_dict to look up emb_set values of images later for the contrastive loss
            emb_set_dict = dict()

            for batch, (image, filenames) in enumerate(val_loader):
                image = image.to(args.device)

                (recon_dvae, cross_entropy, mse, attns, emb_set) = model(image, tau)

                if args.contrastive_loss:
                    video_ids = [fn.split("_")[1] for fn in filenames]
                    label_map = {vid: i for i, vid in enumerate(np.unique(video_ids))}
                    labels = torch.tensor([label_map[vid] for vid in video_ids]).to(
                        args.device
                    )
                    embeddings = torch.flatten(emb_set, start_dim=1)
                
                    xent_loss = ntxent_loss(embeddings, labels)

                    # emb_set = emb_set.to("cpu", non_blocking=True)

                    # for i, filename in enumerate(filenames):
                    #     filename = filename.split("/")[-1]
                    #     _, video_id, _, frame_id = filename.split("_")
                    #     frame_id = frame_id.split(".")[0]
                    #     video_id, frame_id = int(video_id), int(frame_id)

                    #     if video_id not in emb_set_dict.keys():
                    #         emb_set_dict[video_id] = dict()

                    #     emb_set_dict[video_id][frame_id] = emb_set[i, :, :] # 1, H * W, cnn_hidden_size

                if args.use_dp:
                    mse = mse.mean()
                    cross_entropy = cross_entropy.mean()
                    if args.contrastive_loss:
                        xent_loss = xent_loss.mean()

                val_cross_entropy += cross_entropy.item()
                val_mse += mse.item()
                if args.contrastive_loss:
                    val_nt_xent += xent_loss.item()

            # Apply contrastive loss
            # if args.contrastive_loss:
            #     half_batch = args.batch_size // 2

            #     while len(emb_set_dict) > 0:
            #         # get two random unique video ids
            #         vid1, vid2 = np.random.choice(list(emb_set_dict.keys()), 2, replace=False)

            #         # try to get up to 8 embeddings (of a frame) from each video
            #         embedding_keys1 = np.random.choice(list(emb_set_dict[vid1].keys()), min(half_batch, len(emb_set_dict[vid1])), replace=False)
            #         embedding_keys2 = np.random.choice(list(emb_set_dict[vid2].keys()), min(half_batch, len(emb_set_dict[vid2])), replace=False)
            #         emb_sets1 = [emb_set_dict[vid1].pop(key) for key in embedding_keys1]
            #         emb_sets2 = [emb_set_dict[vid2].pop(key) for key in embedding_keys2]
            #         if len(emb_set_dict[vid1]) == 0:
            #             emb_set_dict.pop(vid1)
            #         if len(emb_set_dict[vid2]) == 0:
            #             emb_set_dict.pop(vid2)

            #         labels1 = len(emb_sets1) * [0]
            #         labels2 = len(emb_sets2) * [1]

            #         batch_embeddings = torch.stack(emb_sets1 + emb_sets2, dim=0)
            #         batch_embeddings = torch.flatten(batch_embeddings, start_dim=1).to(args.device)
            #         batch_labels = torch.tensor(labels1 + labels2).to(args.device)

            #         # if there somehow are no labels at all, continue
            #         if batch_labels.size()[0] == 0:
            #             continue

            #         xent_loss = args.contrastive_loss_weight * ntxent_loss(batch_embeddings, batch_labels)
            #         val_nt_xent += xent_loss.item()

            val_cross_entropy /= val_epoch_size
            val_mse /= val_epoch_size

            if args.contrastive_loss:
                val_nt_xent /= val_epoch_size
                val_loss = (
                    val_mse
                    + val_cross_entropy
                    + args.contrastive_loss_weight * val_nt_xent
                )
            else:
                val_loss = val_mse + val_cross_entropy

            writer.add_scalar("VAL/loss", val_loss, epoch + 1)
            writer.add_scalar("VAL/cross_entropy", val_cross_entropy, epoch + 1)
            writer.add_scalar("VAL/mse", val_mse, epoch + 1)

            if args.contrastive_loss:
                writer.add_scalar("VAL/nt_xent", val_nt_xent, epoch + 1)
                print(
                    "====> Epoch: {:3} \t Loss = {:F} \t MSE = {:F} \t CE = {:F} \t NT Xent = {:F}".format(
                        epoch + 1, val_loss, val_mse, val_cross_entropy, val_nt_xent
                    )
                )
            else:
                print(
                    "====> Epoch: {:3} \t Loss = {:F} \t MSE = {:F} \t CE = {:F}".format(
                        epoch + 1, val_loss, val_mse, val_cross_entropy
                    )
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

                torch.save(
                    {
                        "model": (
                            model.module.state_dict()
                            if args.use_dp
                            else model.state_dict()
                        ),
                        "temp": model.module.image_encoder.sysbinder.prototype_memory.attn.temp,
                    },
                    os.path.join(log_dir, "best_model.pt"),
                )

                if 50 <= epoch:
                    recon_tf = (
                        model.module if args.use_dp else model
                    ).reconstruct_autoregressive(image[:8])
                    grid = visualize(image, recon_dvae, recon_tf, attns, N=8)
                    writer.add_image("VAL_recons/epoch={:03}".format(epoch + 1), grid)

            writer.add_scalar("VAL/best_loss", best_val_loss, epoch + 1)

            checkpoint = {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "model": (
                    model.module.state_dict() if args.use_dp else model.state_dict()
                ),
                "optimizer": optimizer.state_dict(),
                "temp": model.module.image_encoder.sysbinder.prototype_memory.attn.temp,
            }

            ckpt_fn = "checkpoint.pt.tar"
            if (epoch) % 50 == 0:
                ckpt_fn = f"checkpoint_{epoch}.pt.tar"

            torch.save(checkpoint, os.path.join(log_dir, ckpt_fn))

            print("====> Best Loss = {:F} @ Epoch {}".format(best_val_loss, best_epoch))

            torch.cuda.empty_cache()

        torch.save(
            model.module.state_dict() if args.use_dp else model.state_dict(),
            os.path.join(log_dir, "last_model.pt"),
        )

        rtpt.step(f"epoch:{int(epoch)}")

    writer.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
