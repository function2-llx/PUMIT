import sys

sys.path.append("third-party")

from collections import OrderedDict
from copy import deepcopy

from tensorboardX import SummaryWriter
from tqdm import trange


import argparse
import os
import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


class Transform3D:
    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):
        if self.mul == "0.5":
            voxel = voxel * 0.5
        elif self.mul == "random":
            voxel = voxel * np.random.uniform()

        return voxel.astype(np.float32)


class UniMiSSClassifier(nn.Module):
    def __init__(self, encoder, n_classes, interpolate):
        super(UniMiSSClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.embed_dims[3], n_classes)
        self.interpolate = interpolate

    def forward(self, x):
        if self.interpolate:
            # x = nnf.pad(x, (2, 2, 0, 0, 0, 0))
            x = nnf.interpolate(x, size=(96, 96, 96), mode="trilinear")
        else:
            x = nnf.pad(x, (2, 2, 2, 2, 2, 2))
        cls_token = self.encoder(x, "3D")[0][4][:, 0, :]
        logits = self.fc(cls_token)
        return logits


class EVAClassifier(nn.Module):
    def __init__(self, encoder, dim, n_classes, interpolate):
        super(ViTClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(dim, n_classes)
        self.interpolate = interpolate

    def forward(self, x):
        if self.interpolate:
            x = nnf.interpolate(x, size=(28, 224, 224), mode="trilinear")
        x = SpatialTensor(x, 0)
        cls_token = self.encoder(x)[:, 0, :]
        logits = self.fc(cls_token)
        return logits


class PUMITClassifier(nn.Module):
    def __init__(self, encoder, dim, n_classes):
        super(PUMITClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = nnf.interpolate(x, size=(160, 160, 160), mode="trilinear")
        x = SpatialTensor(x, 0)
        cls_token = self.encoder(x)[:, 0, :]
        logits = self.fc(cls_token)
        return logits


class SMITClassifier(nn.Module):
    def __init__(self, encoder, n_classes, interpolate):
        super(SMITClassifier, self).__init__()
        self.encoder = encoder
        self.norm = nn.LayerNorm(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(768, n_classes)
        self.interpolate = interpolate

    def forward(self, x):
        if self.interpolate:
            x = nnf.interpolate(x, size=(96, 96, 96), mode="trilinear")
        else:
            x = nnf.pad(x, (2, 2, 2, 2, 2, 2))
        x, outs = self.encoder(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


def test(model, evaluator, data_loader, criterion, device, run, save_folder=None):
    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())

            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


def train(model, train_loader, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = torch.squeeze(targets, 1).long().to(device)
        loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        writer.add_scalar("train_loss_logs", loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()

    epoch_loss = sum(total_loss) / len(total_loss)
    return epoch_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RUN evaluation on MedMNIST3D")

    parser.add_argument(
        "--output_root",
        default="./output",
        help="output root, where to save models",
        type=str,
    )
    parser.add_argument("--data_flag", default="organmnist3d", type=str)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument(
        "--arch",
        default="UniMiSS_tiny",
        choices=["UniMiSS_tiny", "UniMiSS_small", "EVA_02_B", "PUMIT", "SMIT"],
        type=str,
    )
    parser.add_argument("--shape_transform", action="store_true")
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--run", default="1", type=str)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--interpolate", action="store_true")

    args = parser.parse_args()

    args.output_root = os.path.join(
        args.output_root, args.run, args.arch, args.data_flag
    )
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    print("==> Preparing data...")

    info = medmnist.INFO[args.data_flag]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    train_transform = (
        Transform3D(mul="random") if args.shape_transform else Transform3D()
    )
    eval_transform = Transform3D(mul="0.5") if args.shape_transform else Transform3D()
    train_dataset = DataClass(
        split="train",
        transform=train_transform,
        download=False,
        as_rgb=True if "EVA" in args.arch or "PUMIT" in args.arch else False,
    )
    train_dataset_at_eval = DataClass(
        split="train",
        transform=eval_transform,
        download=False,
        as_rgb=True if "EVA" in args.arch or "PUMIT" in args.arch else False,
    )
    val_dataset = DataClass(
        split="val",
        transform=eval_transform,
        download=False,
        as_rgb=True if "EVA" in args.arch or "PUMIT" in args.arch else False,
    )
    test_dataset = DataClass(
        split="test",
        transform=eval_transform,
        download=False,
        as_rgb=True if "EVA" in args.arch or "PUMIT" in args.arch else False,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=True
    )
    train_loader_at_eval = torch.utils.data.DataLoader(
        dataset=train_dataset_at_eval, batch_size=32, shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=32, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=32, shuffle=False
    )

    print(f"==> Building and training model... Model = {args.arch}")

    train_evaluator = medmnist.Evaluator(args.data_flag, "train")
    val_evaluator = medmnist.Evaluator(args.data_flag, "val")
    test_evaluator = medmnist.Evaluator(args.data_flag, "test")

    if "UniMiSS" in args.arch:
        sys.path.append("third-party/UniMiSS")
        import UniMiSS.models.MiT as MiT

        student = MiT.__dict__["model_tiny" if "tiny" in args.arch else "model_small"](
            norm2D="IN2",
            norm3D="IN3",
            act="LeakyReLU",
            ws=False,
            img_size2D=224,
            img_size3D=[16, 96, 96],
            modal_type="MM",
            drop_path_rate=0.1,
        )

        ckpt = torch.load(args.model_path)
        state_dict = OrderedDict()
        for k, v in ckpt["student"].items():
            if k[7:11] != "head":
                state_dict[k[16:]] = v
        student.load_state_dict(state_dict, strict=True)
        encoder = student.transformer
        model = UniMiSSClassifier(encoder, n_classes, args.interpolate)
    elif args.arch == "EVA_02_B":
        from pumt.model import ViT
        from pumt.model.vit import Checkpoint
        from pumt.sac import SpatialTensor

        ckpt = Checkpoint(path=args.model_path, state_dict_key="module")
        encoder = ViT(
            patch_size=14,
            pretrained_pos_embed_shape=(16, 16),
            adaptive_patch_embed=False,
            pos_embed_shape=(2, 2, 2) if not args.interpolate else (2, 16, 16),
            rope_rescale_shape=(-1, 16, 16),
            pretrained_ckpt=ckpt
        )
        # encoder = ViT(patch_size=14, embed_dim=1024, depth=24, num_heads=16, pretrained_pos_embed_shape=(16, 16),
        #            pos_embed_shape=(2, 2, 2) if not args.interpolate else (2, 16, 16), rope_rescale_shape=(-1, 16, 16), adaptive_patch_embed=False)
        model = EVAClassifier(encoder, encoder.embed_dim, n_classes, args.interpolate)
    elif args.arch == "SMIT":
        sys.path.append("third-party/SMIT")

        from SMIT.models.Trans import SwinTransformer_Unetr

        encoder = SwinTransformer_Unetr(
            in_chans=1,
            embed_dim=48,
            depths=(2, 2, 4, 2),
            num_heads=(4, 4, 4, 4),
            mlp_ratio=4,
            pat_merg_rf=4,
            qkv_bias=False,
            drop_rate=0,
            drop_path_rate=0.3,
            ape=False,
            spe=False,
            patch_norm=True,
            use_checkpoint=True if args.interpolate else False,
            out_indices=(0, 1, 2, 3),
            patch_size=2,
            window_size=(4, 4, 4),
        )
        ckpt = torch.load(args.model_path)
        encoder.load_state_dict(ckpt, strict=False)
        model = SMITClassifier(encoder, n_classes, args.interpolate)
    elif args.arch == "PUMIT":
        from pumt.model import ViT
        from pumt.model.vit import Checkpoint
        from pumt.sac import SpatialTensor

        ckpt = Checkpoint(path=args.model_path)
        encoder = ViT(
            patch_size=16,
            pretrained_pos_embed_shape=(16, 16),
            adaptive_patch_embed=False,
            pos_embed_shape=(10, 10, 10),
            pretrained_ckpt=ckpt
        )
        model = PUMITClassifier(encoder, encoder.embed_dim, n_classes)

    device = torch.device("cuda:0")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    params_groups = (
        model.parameters() if args.arch == "SMIT" else get_params_groups(model)
    )
    optimizer = torch.optim.AdamW(params_groups, args.lr)

    gamma = 0.1
    milestones = [0.5 * args.num_epochs, 0.75 * args.num_epochs]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    print(f"==> Starting training... Learning rate = {args.lr}")

    logs = ["loss", "auc", "acc"]
    train_logs = ["train_" + log for log in logs]
    val_logs = ["val_" + log for log in logs]
    test_logs = ["test_" + log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)

    writer = SummaryWriter(
        log_dir=os.path.join(args.output_root, "Tensorboard_Results")
    )

    best_auc = 0
    best_epoch = 0

    global iteration
    iteration = 0

    for epoch in trange(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, writer)

        train_metrics = test(
            model, train_evaluator, train_loader_at_eval, criterion, device, args.arch
        )
        val_metrics = test(
            model, val_evaluator, val_loader, criterion, device, args.arch
        )
        test_metrics = test(
            model, test_evaluator, test_loader, criterion, device, args.arch
        )

        scheduler.step()

        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)

        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)

            print("cur_best_auc:", best_auc)
            print("cur_best_epoch", best_epoch)

    path = os.path.join(args.output_root, "best_model.pth")

    train_metrics = test(
        best_model,
        train_evaluator,
        train_loader_at_eval,
        criterion,
        device,
        args.arch,
        args.output_root,
    )
    val_metrics = test(
        best_model,
        val_evaluator,
        val_loader,
        criterion,
        device,
        args.arch,
        args.output_root,
    )
    test_metrics = test(
        best_model,
        test_evaluator,
        test_loader,
        criterion,
        device,
        args.arch,
        args.output_root,
    )

    train_log = "train  auc: %.5f  acc: %.5f\n" % (train_metrics[1], train_metrics[2])
    val_log = "val  auc: %.5f  acc: %.5f\n" % (val_metrics[1], val_metrics[2])
    test_log = "test  auc: %.5f  acc: %.5f\n" % (test_metrics[1], test_metrics[2])

    log = "%s\n" % (args.data_flag) + train_log + val_log + test_log + "\n"
    print(log)

    with open(
        os.path.join(args.output_root, "%s_log.txt" % (args.data_flag)), "a"
    ) as f:
        f.write(log)

    writer.close()
