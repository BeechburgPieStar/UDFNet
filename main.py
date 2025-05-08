import argparse
import torch.optim
from data_loader import splitTrainTestSet
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from utils.utils import *
from utils.crl_utils import *
from utils.logger import create_logger
from tqdm import tqdm
from main_test import mainqmf
import os
import torch.optim
from models.ClassifierNet import Bottleneck, Netqmf
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import args_parser

args = args_parser.args_parser()

def get_args(parser):
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="log")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--lamb", type=float, default=1)


class CustomDataset(Dataset):
    def __init__(self, HSI, SAR, labels, indexes):
        self.HSI = HSI
        self.SAR = SAR
        self.labels = labels
        self.indexes = indexes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'HSI': self.HSI[idx],
            'SAR': self.SAR[idx],
            'labels': self.labels[idx],
            'indexes': self.indexes[idx]
        }


def prepare_data_loaders(batch_size, test_ratio):
    X_train = torch.load(os.path.join(args.root, args.dataset, 'modified', 'X_train_l.pt'))
    X_train_2 = torch.load(os.path.join(args.root, args.dataset, 'modified', 'X_train_2_l.pt'))
    gt_train1 = torch.load(os.path.join(args.root, args.dataset, 'modified', 'gt_train_l.pt'))

    X_train = X_train.permute(0, 3, 1, 2)
    X_train_2 = X_train_2.permute(0, 3, 1, 2)

    hsi_train, hsi_val, gt_train, gt_val = splitTrainTestSet(X_train, gt_train1, test_ratio, randomState=128)
    sar_train, sar_val, _, _ = splitTrainTestSet(X_train_2, gt_train1, test_ratio, randomState=128)

    indexes_train = torch.arange(len(gt_train))
    indexes_val = torch.arange(len(gt_val))

    batch_train = {
        'HSI': hsi_train,
        'SAR': sar_train,
        'labels': gt_train,
        'indexes': indexes_train
    }

    batch_val = {
        'HSI': hsi_val,
        'SAR': sar_val,
        'labels': gt_val,
        'indexes': indexes_val
    }

    trainset = CustomDataset(batch_train['HSI'], batch_train['SAR'], batch_train['labels'], batch_train['indexes'],)
    valset = CustomDataset(batch_val['HSI'], batch_val['SAR'], batch_val['labels'], batch_val['indexes'])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def rank_loss(confidence, idx, history):
    # make input pair
    rank_input1 = confidence
    rank_input2 = torch.roll(confidence, -1)
    idx2 = torch.roll(idx, -1)

    # calc target, margin
    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

    # ranking loss
    ranking_loss = nn.MarginRankingLoss(margin=0.0)(rank_input1,
                                                    rank_input2,
                                                    -rank_target)

    return ranking_loss

def model_forward_train(i_epoch, model, args, batch, hsi_history, sar_history):
    hsi, sar, tgt = batch['HSI'], batch['SAR'], batch['labels']
    tgt = tgt - 1
    idx = batch['indexes']

    hsi, sar, tgt = hsi.cuda(), sar.cuda(), tgt.cuda()
    hsi_sar_logits, hsi_logits, sar_logits, hsi_conf, sar_conf = model(hsi, sar)


    hsi_clf_loss = nn.CrossEntropyLoss()(hsi_logits, tgt)
    sar_clf_loss = nn.CrossEntropyLoss()(sar_logits, tgt)
    hsi_sar_clf_loss = nn.CrossEntropyLoss()(hsi_sar_logits, tgt)
    clf_loss = hsi_clf_loss + sar_clf_loss + hsi_sar_clf_loss

    hsi_pred = hsi_logits.argmax(dim=1)
    sar_pred = sar_logits.argmax(dim=1)

    hsi_correctness = (hsi_pred == tgt)
    sar_correctness = (sar_pred == tgt)
    hsi_loss = nn.CrossEntropyLoss(reduction='none')(hsi_logits, tgt).detach()
    sar_loss = nn.CrossEntropyLoss(reduction='none')(sar_logits, tgt).detach()

    hsi_rank_loss = rank_loss(hsi_conf, idx, hsi_history)
    sar_rank_loss = rank_loss(sar_conf, idx, sar_history)

    hsi_history.correctness_update(idx, hsi_loss, hsi_conf.squeeze())
    sar_history.correctness_update(idx, sar_loss, sar_conf.squeeze())

    loss = clf_loss + args.lamb * (hsi_rank_loss + sar_rank_loss)

    return loss, hsi_sar_logits, hsi_logits, sar_logits, tgt


def model_forward_eval(i_epoch, model, args, batch):
    hsi, sar, tgt = batch['HSI'], batch['SAR'], batch['labels']

    hsi, sar, tgt = hsi.cuda(), sar.cuda(), tgt.cuda()
    tgt = tgt - 1
    hsi_sar_logits, hsi_logits, sar_logits, hsi_conf, sar_conf = model(hsi, sar)

    hsi_clf_loss = nn.CrossEntropyLoss()(hsi_logits, tgt)
    sar_clf_loss = nn.CrossEntropyLoss()(sar_logits, tgt)
    hsi_sar_clf_loss = nn.CrossEntropyLoss()(hsi_sar_logits, tgt)
    clf_loss = hsi_clf_loss + sar_clf_loss + hsi_sar_clf_loss

    loss = torch.mean(clf_loss)

    return loss, hsi_sar_logits, hsi_logits, sar_logits, tgt



def model_eval(i_epoch, data, model, args, store_preds=False):
    model.eval()
    with torch.no_grad():
        losses, hsi_preds, sar_preds, hsi_sar_preds, tgts = [], [], [], [], []
        for batch in data:
            loss, hsi_sar_logits, hsi_logits, sar_logits, tgt = model_forward_eval(i_epoch, model, args, batch)
            losses.append(loss.item())

            hsi_pred = hsi_logits.argmax(dim=1).cpu().detach().numpy()
            sar_pred = sar_logits.argmax(dim=1).cpu().detach().numpy()
            hsi_sar_pred = hsi_sar_logits.argmax(dim=1).cpu().detach().numpy()

            hsi_preds.append(hsi_pred)
            sar_preds.append(sar_pred)
            hsi_sar_preds.append(hsi_sar_pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}

    tgts = [l for sl in tgts for l in sl]
    hsi_preds = [l for sl in hsi_preds for l in sl]
    sar_preds = [l for sl in sar_preds for l in sl]
    hsi_sar_preds = [l for sl in hsi_sar_preds for l in sl]
    metrics["hsi_acc"] = accuracy_score(tgts, hsi_preds)
    metrics["sar_acc"] = accuracy_score(tgts, sar_preds)
    metrics["hsi_sar_acc"] = accuracy_score(tgts, hsi_sar_preds)
    return metrics


def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name, str(args.seed))
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader = prepare_data_loaders(batch_size=64, test_ratio=0.2)

    model = Netqmf(hsi_channels=63, sar_channels=1, hidden_size=128, block=Bottleneck, num_parallel=2, num_reslayer=2, num_classes=6, bn_threshold=2e-2)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    hsi_history = History(len(train_loader.dataset))
    sar_history = History(len(train_loader.dataset))

    train_losses_history = []
    val_losses_history = []

    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()
        for batch in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {i_epoch + 1}/{args.max_epochs}"):
            loss, hsi_sar_logits, hsi_logits, sar_logits, tgt = model_forward_train(i_epoch, model, args, batch, hsi_history, sar_history)
            if args.gradient_accumulation_steps > 1:
                 loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        train_losses_history.append(np.mean(train_losses))

        model.eval()
        metrics = model_eval(
            np.inf, val_loader, model, args, store_preds=True
        )
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("val", metrics, logger)
        logger.info(
            "{}: Loss: {:.5f} | hsi_acc: {:.5f}, sar_acc: {:.5f}, hsi sar acc: {:.5f}".format(
                "val", metrics["loss"], metrics["hsi_acc"], metrics["sar_acc"], metrics["hsi_sar_acc"]
            )
        )
        tuning_metric = metrics["hsi_sar_acc"]

        val_losses_history.append(metrics["loss"])

        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    np.savetxt(os.path.join(args.savedir, "train_losses.csv"), train_losses_history, delimiter=",")
    np.savetxt(os.path.join(args.savedir, "val_losses.csv"), val_losses_history, delimiter=",")

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    mainqmf(model)

def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()