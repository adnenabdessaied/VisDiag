#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

"""
In this script, we train or encoder decoder networks using a late fusion encoder and a discriminative decoder as
suggested in the visual dialog paper (arXiv: 1611.08669v5)  
"""

import argparse
import pickle
import numpy as np
import os
from datetime import datetime
import logging
from random import randrange

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from tensorboardX import SummaryWriter
from preprocessing.dataset import Visual_Dialog_Dataset
from metrics import SparseGTMetrics, NDCG
from encoders.memory_network import Memory_Network
from encoders.late_fusion import Late_Fusion

from decoders.discriminative_decoder import Disc_Decoder

from encoder_decoder_net import EncoderDecoderNet
from preprocessing.statics import (
    EMBEDDING_DIM,
    VOCAB_SIZE,
    MAX_SENTENCE_LENGTH,
    ANSWER_GT,
    DEC_TYPE
)

from encoders.statics import (
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    DROPOUT,
    IMG_FEATURE_DIM,
)

from tb_fn import decorate_tb_image, get_tb_image_grid
logging.basicConfig(level=logging.INFO)


def _get_current_timestamp() -> str:
    """
    A function that returns the current timestamp to be used to name the checkpoint files of the model.
    :return: The current timestamp.
    """
    current_time = datetime.now()
    current_time = current_time.strftime("%d-%b-%Y_(%H:%M:%S.%f)")
    return current_time


def _set_learning_rate(optimizer, new_lr):
    """
    Updates the learning rate of an optimizer.
    :param optimizer: The optimizer used in training.
    :param new_lr: The new learning rate
    :return: None.
    """
    for param in optimizer.param_groups:
        param["lr"] = new_lr


def _get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-tr",
                            "--tr_json",
                            required=False,
                            default="/data/vis_diag/visdial_0.9_train.json",
                            help="Path to the json file that will be used for training.")

    arg_parser.add_argument("-val",
                            "--val_json",
                            required=False,
                            default="/data/vis_diag/visdial_0.9_val.json",
                            help="Path to the json file that will be used for validation.")

    arg_parser.add_argument("-te",
                            "--test_json",
                            required=False,
                            default="/data/vis_diag/visdial_1.0_test.json",
                            help="Path to the json file that will be used for validation.")

    arg_parser.add_argument("-den",
                            "--dense_json",
                            required=False,
                            default="/data/vis_diag/visdial_1.0_val_dense_annotations.json",
                            help="Path to the dense annotations of the validation data.")

    arg_parser.add_argument("-img",
                            "--images_dir",
                            required=False,
                            default="/data/vis_diag/images",
                            help="Path to the directory containing all the images.")

    arg_parser.add_argument("-w2i",
                            "--word_to_index",
                            required=False,
                            default="/home/adnen/Documents/Studium/research_projects/visual_dialog/visual_dialog/"
                                    "word_to_idx_09.pickle",
                            help="Path to a saved vocabulary.")

    arg_parser.add_argument("-b",
                            "--batch_size",
                            required=False,
                            default=12,
                            help="Batch size.")

    arg_parser.add_argument("-e",
                            "--epochs",
                            required=False,
                            default=20,
                            help="Number of epochs for training.")

    arg_parser.add_argument("-ll",
                            "--lstm_num_layers",
                            required=False,
                            default=2,
                            help="Number of lstm layers.")

    arg_parser.add_argument("-lh",
                            "--lstm_hidden_size",
                            required=False,
                            default=512,
                            help="Lstm hidden size.")

    arg_parser.add_argument("-embdim",
                            "--embedding_dim",
                            required=False,
                            default=300,
                            help="Dimension of the word embedding.")

    arg_parser.add_argument("-d",
                            "--dropout",
                            required=False,
                            default=0.5,
                            help="Dropout.")

    arg_parser.add_argument("-id",
                            "--image_dim",
                            required=False,
                            default=4096,
                            help="Image feature dimension.")

    arg_parser.add_argument("-m",
                            "--max_len",
                            required=False,
                            default=15,
                            help="Maximum sentence length.")

    arg_parser.add_argument("-tb",
                            "--tensorboard",
                            required=False,
                            default="/home/adnen/Documents/Studium/research_projects/visual_dialog/visual_dialog/"
                                    "summaries",
                            help="Tensorboard summaries directory.")

    arg_parser.add_argument("-chkpt",
                            "--checkpoints",
                            required=False,
                            default="/home/adnen/Documents/Studium/research_projects/visual_dialog/visual_dialog/"
                                    "checkpoints",
                            help="Directory for check-pointing the network.")

    arg_parser.add_argument("-bchkpt",
                            "--best_checkpoint",
                            required=False,
                            default="/home/adnen/Documents/Studium/research_projects/visual_dialog/visual_dialog/"
                                    "checkpoints/best",
                            help="Directory for check-pointing the network.")

    args = vars(arg_parser.parse_args())
    return args


def train(args):
    with open(args["word_to_index"], "rb") as f:
        word_to_idx = pickle.load(f)

    # Define params, a dict that stores all the hyper-parameters of the network
    params = {EMBEDDING_DIM: args["embedding_dim"],
              VOCAB_SIZE: len(word_to_idx),
              LSTM_HIDDEN_SIZE: args["lstm_hidden_size"],
              LSTM_NUM_LAYERS: args["lstm_num_layers"],
              DROPOUT: args["dropout"],
              IMG_FEATURE_DIM: args["image_dim"],
              MAX_SENTENCE_LENGTH: args["max_len"],
              DEC_TYPE: "disc"
              }

    # Reproducibility --> https://pytorch.org/docs/stable/notes/randomness
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info("Using {}".format(torch.cuda.get_device_name()))
    else:
        device = torch.device("cpu")
        logging.info("Using the CPU")

    # Prepare train and validation datasets
    logging.info("Loading the data ...")
    tr_dataset = Visual_Dialog_Dataset(args["tr_json"], args["tr_json"], args["images_dir"],
                                       args["word_to_index"],
                                       params, False, True,
                                       dense_annotations_file=args["dense_json"],
                                       concatenate_history=True)

    tr_dialog_reader = tr_dataset.dialog_reader
    val_dialog_reader = tr_dataset.dialog_reader

    # Only 80K of visdial_0.9 data will be used for training as indicated in the paper.
    if "visdial_0.9" in args["tr_json"]:
        split = random_split(tr_dataset, (80000, len(tr_dataset) - 80000))
        tr_dataset, val_dataset = split[0], split[-1]

        test_dataset = Visual_Dialog_Dataset(args["val_json"], args["tr_json"], args["images_dir"],
                                             args["word_to_index"],
                                             params, False, True,
                                             dense_annotations_file=args["dense_json"],
                                             concatenate_history=True)
    else:
        val_dataset = Visual_Dialog_Dataset(args["val_json"], args["tr_json"], args["images_dir"],
                                            args["word_to_index"],
                                            params, False, True,
                                            dense_annotations_file=args["dense_json"],
                                            concatenate_history=True)

        test_dataset = Visual_Dialog_Dataset(args["test_json"], args["tr_json"], args["images_dir"],
                                             args["word_to_index"],
                                             params, False, True,
                                             dense_annotations_file=args["dense_json"],
                                             concatenate_history=True)
        val_dialog_reader = val_dataset.dialog_reader

    # Construct train and validation data loaders
    logging.info("Constructing the data loaders ...")
    tr_data_loader = DataLoader(tr_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=6)
    val_data_loader = DataLoader(val_dataset, batch_size=int(args["batch_size"]), shuffle=True, num_workers=6)
    test_data_loader = DataLoader(test_dataset, batch_size=int(args["batch_size"]), shuffle=True, num_workers=6)

    logging.info("Data loaders successfully constructed ...")

    # We use nn.CrossEntropyLoss() since our decoder is discriminative.
    criterion = nn.CrossEntropyLoss()

    # Load the most recent checkpoint. Otherwise start training from scratch.
    checkpoints = [ckpt for ckpt in os.listdir(args["checkpoints"]) if ckpt.endswith("pth")]
    checkpoints = [os.path.join(args["checkpoints"], checkpoint) for checkpoint in checkpoints]
    if len(checkpoints) > 0:
        most_recent_chkpt_path = max(checkpoints, key=os.path.getctime)
        most_recent_chkpt = torch.load(most_recent_chkpt_path)
        net = most_recent_chkpt["net"]
        net.load_state_dict(most_recent_chkpt["net_state_dict"])

        # We want to train further
        net.train()

        # Send the net first to the device to avoid potential runtime errors caused by the optimizer if we resume
        # training on a different device
        net.to(device)

        optimizer = most_recent_chkpt["optimizer"]
        optimizer.load_state_dict(most_recent_chkpt["optimizer_state_dict"])

        start_epoch = most_recent_chkpt["epoch"]
        lr = most_recent_chkpt["lr"]
        batch_iter_tr = most_recent_chkpt["batch_iter_tr"]
        batch_iter_val = most_recent_chkpt["batch_iter_val"]
        chkpt_timestamp = os.path.getmtime(most_recent_chkpt_path)
        logging.info("Network loaded from the latest checkpoint saved on {}".format(datetime.fromtimestamp(
            chkpt_timestamp)))

    else:
        # Construct the late fusion encoder and discriminative decoder
        encoder = Late_Fusion(params)
        decoder = Disc_Decoder(params)

        # Share word embedding between encoder and decoder.
        decoder.embedding = encoder.embedding

        # Construct the encoder decoder net
        net = EncoderDecoderNet(encoder, decoder).to(device)
        logging.info("Encoder-decoder network successfully constructed...")

        optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=True)
        lr = 1e-3
        start_epoch = 0
        batch_iter_tr = 0
        batch_iter_val = 0

    # Define the summary writer to be used for tensorboard visualizations.
    summary_writer = SummaryWriter(log_dir=args["tensorboard"])

    # Define the metrics
    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()

    modes = ["train", "val"]

    best_loss_val = 0
    for epoch in range(start_epoch + 1, args["epochs"] + 1):
        # Update the lr
        lr *= 0.7
        summary_writer.add_scalar("Learning_rate", lr, epoch)
        _set_learning_rate(optimizer, lr)
        for mode in modes:
            if mode == "train":
                pbar_train = tqdm(tr_data_loader)
                pbar_train.set_description("{} | Epoch {} / {}".format(mode, epoch, args["epochs"]))
                for batch, image_paths, image_ids in pbar_train:
                    # Send the data to the appropriate device
                    batch = dict(zip(batch.keys(), map(lambda x: x.to(device), batch.values())))

                    optimizer.zero_grad()
                    output = net(batch)

                    # gt of the whole batch
                    # Shape: (batch_size * rounds)
                    gt = batch[ANSWER_GT].view(-1)
                    sparse_metrics.observe(output, batch[ANSWER_GT])

                    # Shape: (batch_size * rounds, answer_opts)
                    output = output.view(-1, output.size(-1))

                    # Randomly select an image of the batch
                    rand_idx = randrange(len(image_ids))

                    # Get the decorated images and plot them on tensorboard
                    img_grid = get_tb_image_grid(image_paths[rand_idx], gt, output, rand_idx, [0, 4, 9])
                    summary_writer.add_images(mode, img_grid, global_step=batch_iter_tr, dataformats="NHWC")

                    for i in range(1):
                        decorated_image = decorate_tb_image(
                            image_paths[rand_idx], gt, output, rand_idx, i + 1, image_ids[rand_idx].item(),
                            tr_dialog_reader)
                        summary_writer.add_image(mode + "_{}_pred".format(i + 1), decorated_image,
                                                 global_step=batch_iter_tr, dataformats="HWC")

                    # Compute the training loss
                    tr_loss = criterion(output, gt)

                    # Back propagation with anomaly detection -> Makes it easier to locate the faulty parts of the net
                    # if some undesirable phenomena happen, e.g. if some layers produce NaN of Inf values.
                    with torch.autograd.detect_anomaly():
                        tr_loss.backward()

                    # Clamp the gradients to avoid explosion
                    nn.utils.clip_grad_norm_(net.parameters(), 5)

                    optimizer.step()

                    # Compute the training metrics & update the TB summaries
                    summary_writer.add_scalar("Training_loss", tr_loss, batch_iter_tr)
                    tr_metrics = {}
                    tr_metrics.update(sparse_metrics.retrieve(reset=True))
                    summary_writer.add_scalars(mode + "_metrics", tr_metrics, batch_iter_tr)
                    batch_iter_tr += 1

                    # Release GPU cache memory
                    torch.cuda.empty_cache()

                timestamp = _get_current_timestamp()
                torch.save({
                    "net": net,
                    "net_state_dict": net.state_dict(),
                    "optimizer": optimizer,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "lr": lr,
                    "batch_iter_tr": batch_iter_tr,
                    "batch_iter_val": batch_iter_val,
                    }, os.path.join(args["checkpoints"], "checkpoint_{}_{}.pth".format(timestamp, epoch)))

                # Delete the oldest checkpoint if the number of checkpoints exceeds 10 to save disk space.
                checkpoints = [ckpt for ckpt in os.listdir(args["checkpoints"]) if ckpt.endswith("pth")]
                checkpoints = [os.path.join(args["checkpoints"], checkpoint) for checkpoint in checkpoints]
                if len(checkpoints) > 10:
                    oldest_checkpoint_pth = min(checkpoints, key=os.path.getctime)
                    os.remove(oldest_checkpoint_pth)

            else:
                # Set the net to eval mode
                net.eval()
                pbar_val = tqdm(val_data_loader)
                pbar_val.set_description("{} | Epoch {} / {}".format(mode, epoch, args["epochs"]))
                val_loss_epoch = 0
                for batch, image_paths, image_ids in pbar_val:

                    # Send the data to the appropriate device
                    batch = dict(zip(batch.keys(), map(lambda x: x.to(device), batch.values())))

                    with torch.no_grad():
                        # Shape :(batch_size, num_rounds, num_opts)
                        output = net(batch)

                    # gt of the whole batch
                    # Shape: (batch_size * rounds)
                    gt = batch[ANSWER_GT].view(-1)
                    sparse_metrics.observe(output, batch[ANSWER_GT])

                    # Only visdial_1.0 has dense annotations
                    if "visdial_1.0" in args["tr_json"]:
                        output_ndgc = output[torch.arange(output.size(0)), batch["round_id"] - 1, :]
                        ndcg.observe(output_ndgc, batch["gt_relevance"])
                    # Shape: (batch_size * rounds, answer_opts)
                    output = output.view(-1, output.size(-1))

                    # Randomly select an image of the batch
                    rand_idx = randrange(len(image_ids))

                    # Get the decorated images and show them on tensorboard
                    img_grid = get_tb_image_grid(image_paths[rand_idx], gt, output, rand_idx, [0, 4, 9])
                    summary_writer.add_images(mode, img_grid, global_step=batch_iter_val, dataformats="NHWC")

                    for i in range(1):
                        decorated_image = decorate_tb_image(
                            image_paths[rand_idx], gt, output, rand_idx, i + 1, image_ids[rand_idx].item(),
                            val_dialog_reader)
                        summary_writer.add_image(mode + "_{}_pred".format(i + 1), decorated_image,
                                                 global_step=batch_iter_val, dataformats="HWC")

                    # Compute the validation loss
                    val_loss = criterion(output, gt)

                    # Update the val_loss_epoch
                    val_loss_epoch += val_loss

                    # Update the TB summaries
                    summary_writer.add_scalar("Validation_loss", val_loss, batch_iter_val)
                    val_metrics = {}
                    val_metrics.update(sparse_metrics.retrieve(reset=True))
                    val_metrics.update(ndcg.retrieve(reset=True))
                    summary_writer.add_scalars(mode + "_metrics", val_metrics, batch_iter_val)
                    batch_iter_val += 1

                    # Release GPU cache memory
                    torch.cuda.empty_cache()

                # Save the best net, i.e. the net that scores the best loss function on the validation set.
                if epoch == 1:
                    timestamp = _get_current_timestamp()
                    torch.save({
                        "net": net,
                        "net_state_dict": net.state_dict(),
                        "optimizer": optimizer,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "lr": lr,
                        "batch_iter_tr": batch_iter_tr,
                        "batch_iter_val": batch_iter_val,
                    }, os.path.join(args["best_checkpoint"], "checkpoint_{}_{}.pth".format(timestamp, epoch)))
                    best_loss_val = val_loss_epoch
                elif val_loss_epoch <= best_loss_val:
                    # Delete the older
                    checkpoints = os.listdir(args["best_checkpoint"])
                    checkpoints = [os.path.join(args["best_checkpoint"], checkpoint) for checkpoint in checkpoints]
                    [os.remove(checkpoint) for checkpoint in checkpoints]

                    timestamp = _get_current_timestamp()
                    torch.save({
                        "net": net,
                        "net_state_dict": net.state_dict(),
                        "optimizer": optimizer,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "lr": lr,
                        "batch_iter_tr": batch_iter_tr,
                        "batch_iter_val": batch_iter_val,
                    }, os.path.join(args["best_checkpoint"], "checkpoint_{}_{}.pth".format(timestamp, epoch)))
                    best_loss_val = val_loss_epoch

                # Switch back to train mode
                net.train()

    # Now we evaluate the best model on the test data
    checkpoints = [ckpt for ckpt in os.listdir(args["best_checkpoint"]) if ckpt.endswith("pth")]
    most_recent_chkpt_path = [os.path.join(args["checkpoints"], checkpoint) for checkpoint in checkpoints][0]
    most_recent_chkpt = torch.load(most_recent_chkpt_path)
    net = most_recent_chkpt["net"]
    net.load_state_dict(most_recent_chkpt["net_state_dict"])
    net.eval()
    pbar_test = tqdm(test_data_loader)
    pbar_test.set_description("{}".format("testing"))
    batch_iter_test = 0

    for batch, image_paths, image_ids in pbar_test:

        # Send the data to the appropriate device
        batch = dict(zip(batch.keys(), map(lambda x: x.to(device), batch.values())))

        with torch.no_grad():
            # Shape :(batch_size, num_rounds, num_opts)
            output = net(batch)

        # gt of the whole batch
        # Shape: (batch_size * rounds)
        gt = batch[ANSWER_GT].view(-1)
        sparse_metrics.observe(output, batch[ANSWER_GT])

        # Only visdial_1.0 has dense annotations
        if "visdial_1.0" in args["tr_json"]:
            output_ndgc = output[torch.arange(output.size(0)), batch["round_id"] - 1, :]
            ndcg.observe(output_ndgc, batch["gt_relevance"])

        # Shape: (batch_size * rounds, answer_opts)
        output = output.view(-1, output.size(-1))
        # Randomly select an image of the batch
        rand_idx = randrange(len(image_ids))

        # Get the decorated images and plot them on tensorboard
        img_grid = get_tb_image_grid(image_paths[rand_idx], gt, output, rand_idx, [0, 4, 9])

        summary_writer.add_images("testing", img_grid, global_step=batch_iter_tr, dataformats="NHWC")
        decorated_image = decorate_tb_image(image_paths[rand_idx], gt, output, rand_idx, 1,
                                            image_ids[rand_idx].item(), test_dataset.dialog_reader)
        summary_writer.add_image("pred_1_test", decorated_image, global_step=batch_iter_test,
                                 dataformats="HWC")

        test_metrics = {}
        test_metrics.update(sparse_metrics.retrieve(reset=True))
        test_metrics.update(ndcg.retrieve(reset=True))
        summary_writer.add_scalars("test_metrics", test_metrics, batch_iter_test)
        batch_iter_test += 1

        # Release GPU memory cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = _get_args()
    train(args)
