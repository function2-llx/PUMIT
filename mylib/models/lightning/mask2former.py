import itertools as it

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as nnf
from transformers.models.mask2former.modeling_mask2former import pair_wise_sigmoid_cross_entropy_loss, pair_wise_dice_loss

from mylib.utils import DataKey
from mylib.conf import Mask2FormerConf
from .model_base import ExpModelBase
from ..transformer_decoder.masked_attention import MaskedAttentionDecoder

def sample_point(feature_map: torch.Tensor, point_coordinates: torch.Tensor, transformed: bool = False, **kwargs) -> torch.Tensor:
    spatial_dims = point_coordinates.shape[-1]
    assert (dim_diff := feature_map.ndim - spatial_dims) in [1, 2]
    batched = dim_diff == 2
    if not batched:
        feature_map = feature_map[None]
        point_coordinates = point_coordinates[None]
    if not transformed:
        point_coordinates = 2 * point_coordinates - 1
    for _ in range(add_dims := spatial_dims + 2 - point_coordinates.ndim):
        point_coordinates = point_coordinates[:, None]
    point_features = nnf.grid_sample(feature_map, point_coordinates, **kwargs)
    for _ in range(add_dims):
        point_features = point_features[:, :, 0]
    if not batched:
        point_features = point_features[0]
    return point_features

# copy
def dice_loss(logits: torch.Tensor, labels: torch.Tensor):
    probs = logits.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.mean()
    return loss

class Mask2Former(ExpModelBase):
    conf: Mask2FormerConf
    class_loss_weight: torch.Tensor

    def __init__(self, conf: Mask2FormerConf):
        super().__init__(conf)

        self.pixel_decoder = self.create_pixel_decoder()
        self.transformer_decoder = self.create_transformer_decoder()

        if conf.print_shape:
            dummy_input, dummy_output = self.backbone_dummy()
            feature_maps = self.pixel_decoder.forward(dummy_output.feature_maps, dummy_input).feature_maps
            print('pixel decoder output shapes:')
            for feature_map in feature_maps:
                print(feature_map.shape[1:])
        self.class_predictor = nn.Linear(self.transformer_decoder.hidden_dim, conf.num_fg_classes + 1)
        self.register_buffer('class_loss_weight', torch.tensor([conf.eos_coef, *it.repeat(1, conf.num_fg_classes)]))

    def create_pixel_decoder(self):
        return create_model(self.conf.pixel_decoder, decoder_registry)

    def create_transformer_decoder(self) -> MaskedAttentionDecoder:
        return create_model(self.conf.transformer_decoder, transformer_decoder_registry)

    def forward_mask(self, x, manual_mask):
        backbone_feature_maps = self.backbone.forward(x).feature_maps
        feature_maps = self.pixel_decoder.forward(backbone_feature_maps, x).feature_maps
        layers_mask_embeddings, layers_mask_logits = self.transformer_decoder.forward(
            feature_maps[1:],
            feature_maps[0],
            manual_mask,
        )
        return layers_mask_embeddings, layers_mask_logits

    def forward(self, x: torch.Tensor, manual_mask: torch.Tensor | None = None):
        layers_mask_embeddings, layers_mask_logits = self.forward_mask(x, manual_mask)
        layers_class_logits = [
            self.class_predictor(mask_embeddings)
            for mask_embeddings in layers_mask_embeddings
        ]
        return layers_class_logits, layers_mask_logits

    @torch.no_grad()
    def match(self, class_logits: torch.Tensor, mask_logits: torch.Tensor, class_labels: list[torch.Tensor], mask_labels: list[torch.Tensor]):
        batch_size = class_logits.shape[0]
        conf = self.conf
        indices = []
        for i in range(batch_size):
            if class_labels[i].numel() == 0:
                indices.append((np.empty(0), np.empty(0)))
            else:
                class_pred_probs = class_logits[i].softmax(-1)
                # Compute the classification cost. Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class]. The 1 is a constant that doesn't change the matching, it can be ommitted.
                cost_class = -class_pred_probs[:, class_labels[i]]
                # Sample the same set of points for ground truth and predicted masks
                point_coordinates = torch.rand(conf.num_train_points, conf.spatial_dims, device=mask_logits.device)
                point_coordinates = 2 * point_coordinates - 1
                sampled_mask_logits = sample_point(mask_logits[i], point_coordinates, transformed=True)
                sampled_mask_labels = sample_point(mask_labels[i], point_coordinates, transformed=True)

                # compute the cross entropy loss between each mask pairs -> shape (num_queries, num_labels)
                with torch.autocast(device_type=sampled_mask_logits.device.type, enabled=False):
                    cost_bce = pair_wise_sigmoid_cross_entropy_loss(sampled_mask_logits, sampled_mask_labels)
                    # Compute the dice loss betwen each mask pairs -> shape (num_queries, num_labels)
                    cost_dice = pair_wise_dice_loss(sampled_mask_logits, sampled_mask_labels)
                # final cost matrix
                cost_matrix = conf.cost_bce * cost_bce + conf.cost_class * cost_class + conf.cost_dice * cost_dice
                # do the assigmented using the hungarian algorithm in scipy
                assigned_indices: tuple[np.ndarray, np.ndarray] = linear_sum_assignment(cost_matrix.cpu())
                indices.append(assigned_indices)

        return indices

    # logits: nq * 1 * (*sp)
    def sample_points_using_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        conf = self.conf
        num_masks = logits.shape[0]
        num_points_sampled = int(conf.num_train_points * conf.oversample_ratio)

        # Get different sets of random point coordinates
        point_coordinates = torch.rand(num_masks, num_points_sampled, conf.spatial_dims, device=logits.device)
        # Get sampled prediction value for the point coordinates
        point_logits = sample_point(logits, point_coordinates)[:, 0]
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = -torch.abs(point_logits)

        num_uncertain_points = int(conf.importance_sample_ratio * conf.num_train_points)
        num_random_points = conf.num_train_points - num_uncertain_points

        _, uncertain_idx = torch.topk(point_uncertainties, k=num_uncertain_points, dim=1)
        point_coordinates = point_coordinates[
            torch.arange(uncertain_idx.shape[0], device=uncertain_idx.device)[:, None],
            uncertain_idx,
        ]
        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_masks, num_random_points, conf.spatial_dims, device=logits.device)],
                dim=1,
            )
        return point_coordinates

    def cal_class_loss(self, class_logits: torch.Tensor, class_labels: list[torch.Tensor], indices: list[tuple[np.ndarray, np.ndarray]]):
        target_classes = class_logits.new_zeros(class_logits.shape[:2], dtype=torch.long)
        for i, (pred_idx, label_idx) in enumerate(indices):
            target_classes[i, pred_idx] = class_labels[i][label_idx]
        return {
            'class': nnf.cross_entropy(
                class_logits.view(-1, class_logits.shape[-1]),
                target_classes.view(-1),
                self.class_loss_weight,
            )
        }

    def cal_mask_loss(self, mask_logits: torch.Tensor, mask_labels: list[torch.Tensor], indices: list[tuple[np.ndarray, np.ndarray]]):
        matched_mask_logits = []
        matched_mask_labels = []
        for i, (pred_idx, label_idx) in enumerate(indices):
            matched_mask_logits.append(mask_logits[i, pred_idx])
            matched_mask_labels.append(mask_labels[i][label_idx])
        matched_mask_logits = torch.cat(matched_mask_logits, dim=0)[:, None]
        matched_mask_labels = torch.cat(matched_mask_labels, dim=0)[:, None]
        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(matched_mask_logits)
            point_coordinates = 2 * point_coordinates - 1
            point_labels = sample_point(matched_mask_labels, point_coordinates, transformed=True)[:, 0]
        point_logits = sample_point(matched_mask_logits, point_coordinates, transformed=True)[:, 0]
        with torch.autocast(device_type=point_logits.device.type, enabled=False):
            return {
                'bce': nnf.binary_cross_entropy_with_logits(point_logits, point_labels),
                'dice': dice_loss(point_logits, point_labels),
            }

    def cal_loss(self, class_logits: torch.Tensor, mask_logits: torch.Tensor, class_labels: list[torch.Tensor], mask_labels: list[torch.Tensor]):
        indices = self.match(class_logits, mask_logits, class_labels, mask_labels)
        return {
            **self.cal_class_loss(class_logits, class_labels, indices),
            **self.cal_mask_loss(mask_logits, mask_labels, indices),
        }

    def combine_loss(self, loss_dict: dict):
        return torch.stack([
            v * getattr(self.conf, f'cost_{k}')
            for k, v in loss_dict.items()
        ]).sum()

    def training_step(self, batch):
        layers_class_logits, layers_mask_logits = self.forward(batch[DataKey.IMG])
        class_labels = batch[DataKey.CLS]
        mask_labels = batch[DataKey.SEG]
        loss_dicts = [
            self.cal_loss(class_logits, mask_logits, class_labels, mask_labels)
            for class_logits, mask_logits in zip(layers_class_logits, layers_mask_logits)
        ]
        for i in self.conf.log_layers:
            for k, v in loss_dicts[i].items():
                self.log(f'train/layer-{i}/{k}', v)
        loss = torch.stack([self.combine_loss(loss_dict) for loss_dict in loss_dicts]).mean()
        self.log('train/loss', loss)
        return loss

def test():
    from omegaconf import OmegaConf
    from mylib.conf import ModelConf
    conf: Mask2FormerConf = OmegaConf.structured(Mask2FormerConf)
    conf.num_input_channels = 3
    conf.sample_shape = (16, 192, 256)
    conf.num_fg_classes = 2
    conf.num_train_points = 98304
    conf.backbone = ModelConf(name='swin3d', kwargs={
        'in_channels': 3,
        'patch_size': (1, 4, 4),
        'layer_channels': [128, 256, 512, 1024],
        'window_sizes': [(2, 6, 8), (4, 6, 8), (4, 6, 8), (4, 6, 8)],
        'layer_depths': [2, 2, 6, 2],
        'strides': [(1, 2, 2), (2, 2, 2), (2, 2, 2)],
        'num_heads': [4, 8, 16, 32],
    })
    conf.pixel_decoder = ModelConf(name='msdeform', kwargs={
        'spatial_dims': 3,
        'backbone_feature_channels': [128, 256, 512, 1024],
        'feature_dim': 256,
        'num_heads': 8,
    })
    conf.transformer_decoder = ModelConf(name='mask', kwargs={'spatial_dims': 3, 'feature_channels': 256})
    conf.print_shape = False
    model = Mask2Former(conf).cuda()
    batch_size = 4
    batch = {
        DataKey.IMG: torch.randn(batch_size, 3, 16, 192, 256).cuda(),
        DataKey.CLS: [
            torch.empty(0, dtype=torch.long).cuda(),
            torch.tensor([1]).cuda(),
            torch.tensor([2]).cuda(),
            torch.tensor([1, 2]).cuda(),
        ],
        DataKey.SEG: [
            torch.randint(2, size=(0, 16, 192, 256)).float().cuda(),
            torch.randint(2, size=(1, 16, 192, 256)).float().cuda(),
            torch.randint(2, size=(1, 16, 192, 256)).float().cuda(),
            torch.randint(2, size=(2, 16, 192, 256)).float().cuda(),
        ]
    }
    model.training_step(batch)
