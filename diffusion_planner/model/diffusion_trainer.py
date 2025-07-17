import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.normalizer import StateNormalizer

# from .loss.esdf_collision_loss import ESDFCollisionLoss
from pytorch_lightning.utilities import grad_norm
# from diffusion_planner.utils.min_norm_solvers import MinNormSolver

logger = logging.getLogger(__name__)

# torch.set_float32_matmul_precision('high')
class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        init_weights: list[float] = [1, 1],
        diffusion_model_type: str = 'x_start',
        ddp:bool=True
    ) -> None:
        """
        初始化 Lightning 模块.
        
        :param model: 你的 PyTorch 模型.
        :param lr: 学习率.
        :param weight_decay: 权重衰减.
        # ... 其他参数 ...
        :param diffusion_model_type: 扩散模型类型.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        # 将依赖项保存为类属性
        self.diffusion_model_type = diffusion_model_type
        self.ddp = ddp
        init_weights = [float(w) for w in init_weights]
        self.weights = torch.tensor(init_weights, dtype=torch.float32)
        self.weights = self.weights.to(self.device)
    def on_fit_start(self) -> None:
        # pass
        metrics_collection = MetricCollection(
            [
                # minADE().to(self.device),
                # minFDE().to(self.device),
                # MR(miss_threshold=2).to(self.device),
                # PredAvgADE().to(self.device),
                # PredAvgFDE().to(self.device),
            ]
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        # print(f"LightingTrainer _step _step _step _step _step")
        
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch
        data = features["feature"].data
        # ===================================================================
        #  只有在训练阶段才计算和返回损失
        # ===================================================================
        if self.training:
            '''
            data["neighbor_agents_past"]:torch.Size([1, 32, 21, 11]),dtype=torch.float32,device:cuda:0
            data["neighbor_agents_future"]:torch.Size([1, 32, 80, 3]),dtype=torch.float32,device:cuda:0
            data["static_objects"]:torch.Size([1, 5, 10]),dtype=torch.float32,device:cuda:0
            data["lanes"]:torch.Size([1, 70, 20, 12]),dtype=torch.float32,device:cuda:0
            data["lanes_speed_limit"]:torch.Size([1, 70, 1]),dtype=torch.float32,device:cuda:0
            data["lanes_has_speed_limit"]:torch.Size([1, 70, 1]),dtype=torch.bool,device:cuda:0
            data["route_lanes"]:torch.Size([1, 25, 20, 12]),dtype=torch.float32,device:cuda:0
            data["route_lanes_speed_limit"]:torch.Size([1, 25, 1]),dtype=torch.float32,device:cuda:0
            data["route_lanes_has_speed_limit"]:torch.Size([1, 25, 1]),dtype=torch.bool,device:cuda:0
            data["ego_current_state"]:torch.Size([1, 10]),dtype=torch.float32,device:cuda:0
            data["ego_agent_future"]:torch.Size([1, 80, 3]),dtype=torch.float32,device:cuda:0
            '''
            
            # for k,v in data.items():
            #     print(f"model imput:{k}:{v.shape},dtype={v.dtype},device:{v.device}")

            inputs = {
                'ego_current_state': data["ego_current_state"],
                'neighbor_agents_past': data["neighbor_agents_past"],
                'lanes': data["lanes"],
                'lanes_speed_limit': data["lanes_speed_limit"],
                'lanes_has_speed_limit': data["lanes_has_speed_limit"],
                'route_lanes': data["route_lanes"],
                'route_lanes_speed_limit': data["route_lanes_speed_limit"],
                'route_lanes_has_speed_limit': data["route_lanes_has_speed_limit"],
                'static_objects': data["static_objects"]
            }
            # heading to cos sin
            
            
            ego_future = data["ego_agent_future"]
            ego_future = torch.cat(
                [
                    ego_future[..., :2],
                    torch.stack(
                        [ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1
                    ),
                ],
                dim=-1,
                ) 
            neighbors_future = data["neighbor_agents_future"] 
            neighbor_future_mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
            neighbors_future = torch.cat(
            [
                neighbors_future[..., :2],
                torch.stack(
                    [neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )
            neighbors_future[neighbor_future_mask] = 0.

            inputs = self.model.config.observation_normalizer(inputs)
            losses = {}

            losses, _ = diffusion_loss_func(
                self.model,
                inputs,
                self.model.sde.marginal_prob,
                (ego_future, neighbors_future, neighbor_future_mask),
                self.model.config.state_normalizer,
                losses,
                self.model.config.diffusion_model_type
            )
            losses["loss"]=(
                losses["ego_planning_loss"]*self.weights[0]
                + losses["neighbor_prediction_loss"]*self.weights[1]
            )
            metrics = {}
            self._log_step(losses["loss"], losses, metrics, prefix)
        # ===================================================================
        #  在验证/测试阶段，只进行前向推理和指标计算
        # ===================================================================
        else: # not self.training
            # 在验证/测试时，我们不需要构造扩散过程的输入，
            # 直接使用原始的 inputs 字典进行推理即可。
            inputs = data # 或者您可以根据模型 forward 的需要构建
            
            # 直接调用模型进行前向推理
            # self.model.forward 会调用内部 Decoder 的 forward，
            # 因为是 eval 模式，会返回 {'prediction': ...}
            predictions = self.forward(inputs) 
            
            # TODO: 在这里计算您的验证指标 (validation metrics)
            # 例如: ADE, FDE, Collision Rate 等
            # val_metrics = self._compute_metrics(predictions, targets)
            # self.log_dict(val_metrics)
            
            # 验证步骤通常不返回损失，或者返回一个象征性的0
            losses = {
                "loss": 0,
                "ego_planning_loss": 0,
                "neighbor_prediction_loss": 0,
            }
            metrics = {}
            self._log_step(losses["loss"], losses, metrics, prefix)
            return torch.tensor(0.0, device=self.device)
    
    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        
        loss = (
            
        )
        if self.training:
            self.losses = []

        return {
            "loss": loss,
        }

    def get_prediction_loss(self, data, prediction, valid_mask, target):
        """
        prediction: (bs, A-1, T, 6)
        valid_mask: (bs, A-1, T)
        target: (bs, A-1, 6)
        """
        # 先计算一下有效点的个数
        num_valid = valid_mask.sum()
        
        # 如果没有任何有效点，就直接返回 0 loss
        if num_valid == 0:
            # 保证返回值是一个在同一设备、同一 dtype 下的标量张量
            return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
        # print(f"prediction.shape:{prediction[valid_mask].shape}")
        # print(f"target.shape:{target[valid_mask].shape}")
        prediction_loss = F.smooth_l1_loss(
            prediction[valid_mask], target[valid_mask], reduction="none"
        ).sum(-1)
        eps = 1e-6 
        prediction_loss = prediction_loss.sum() / (valid_mask.sum()+eps)
        # print(f"prediction_loss={prediction_loss}")
        # print(f"valid_mask.sum()={valid_mask.sum()}")
        return prediction_loss

    def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
        """
        trajectory: (bs, R, M, T, 4)
        valid_mask: (bs, T)
        """
        num_valid_points = valid_mask.sum(-1)
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s
        r_padding_mask = ~data["reference_line"]["valid_mask"][:bs].any(-1)  # (bs, R)
        future_projection = data["reference_line"]["future_projection"][:bs][
            torch.arange(bs), :, endpoint_index
        ]

        target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        )
        target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0] / self.mode_interval
        ).long()
        target_m_index.clamp_(min=0, max=self.num_modes - 1)

        target_label = torch.zeros_like(probability)
        target_label[torch.arange(bs), target_r_index, target_m_index] = 1

        best_trajectory = trajectory[torch.arange(bs), target_r_index, target_m_index]

        if self.use_collision_loss:
            collision_loss = self.collision_loss(
                best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision_loss = trajectory.new_zeros(1)

        reg_loss = F.smooth_l1_loss(best_trajectory, target, reduction="none").sum(-1)
        # print(f"best_trajectory.shape:{best_trajectory.shape}")
        # print(f"target.shape:{target.shape}")
        # print(f"reg_loss.shape:{reg_loss.shape}")
        # print(f"target:{target}")
        # print(f"valid_mask:{valid_mask}")
        if self.time_decay:
            # decay_weights = torch.exp(-(0.1*torch.arange(reg_loss.shape[-1], device=reg_loss.device))**2).unsqueeze(0)
            decay_weights = torch.exp(-0.1*torch.arange(reg_loss.shape[-1], device=reg_loss.device) 
                                        / torch.exp(torch.tensor(1, device=reg_loss.device))).unsqueeze(0)
            reg_loss = decay_weights*reg_loss/decay_weights.mean()
        if self.time_norm:
            reg_loss = reg_loss/(reg_loss.mean(dim=0, keepdim=True)+1e-6).clone().detach()
        reg_loss = (reg_loss * valid_mask).sum() / valid_mask.sum()

        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        cls_loss = F.cross_entropy(
            probability.reshape(bs, -1), target_label.reshape(bs, -1).detach()
        )

        if self.regulate_yaw:
            heading_vec_norm = torch.norm(best_trajectory[..., 2:4], dim=-1)
            yaw_regularization_loss = F.l1_loss(
                heading_vec_norm, heading_vec_norm.new_ones(heading_vec_norm.shape)
            )
            reg_loss += yaw_regularization_loss
        # 增加与参考线的偏差损失
        selected_reference_lane_xy = data["reference_line"]["position"][:bs][torch.arange(bs), target_r_index][..., :2]
        lane_dev_loss = self._compute_lateral_error(
            best_trajectory[..., :2],
            selected_reference_lane_xy,
            valid_mask
        )

        smoothness_loss = self._batch_angle_smoothness_cost(best_trajectory[..., :2])
        smoothness_loss = (smoothness_loss.squeeze(-1) * valid_mask).sum() / valid_mask.sum()
        # print(f"valid_mask.shape:{valid_mask.shape}")
        # print(f"selected_reference_lane_xy.shape:{selected_reference_lane_xy.shape}")
        # print(f"lane_dev_loss.shape:{lane_dev_loss.shape}")
        return reg_loss, cls_loss, collision_loss, lane_dev_loss, smoothness_loss

    def _compute_lateral_error(
            self,
            pred_xy: torch.Tensor,      # [B, T, 2]
            ref_xy: torch.Tensor,       # [B, T_ref, 2]
            valid_mask: torch.Tensor,   # [B, T] (bool)
            tau: float = 0.1
    ) -> float:
        """
        Differentiable Huber‑smoothed lateral error with valid‑mask support.
        Returns [B, T, 1].
        """
        B, T, _   = pred_xy.shape
        _, T_ref, _ = ref_xy.shape
        dev = pred_xy.device

        # ---------- 1. 参考点的单位法向 ----------
        diff = ref_xy[:, 1:] - ref_xy[:, :-1]
        fwd = F.pad(diff, (0,0,0,1))
        bwd = F.pad(ref_xy[:, 1:] - ref_xy[:, :-1],  (0, 0, 1, 0))
        idx = torch.arange(T_ref, device=ref_xy.device)
        tan = torch.where(idx.eq(0).view(1,-1,1), fwd,
           torch.where(idx.eq(T_ref-1).view(1,-1,1), bwd, fwd+bwd))
        tan = F.normalize(tan, dim=-1, eps=1e-8)
        nor = torch.stack([-tan[...,1], tan[...,0]], dim=-1)           # [B,T_ref,2]

        # ---------- 2. soft‑nearest ----------
        delta      = pred_xy.unsqueeze(2) - ref_xy.unsqueeze(1)   # [B,T,T_ref,2]
        dist_sq    = (delta**2).sum(-1)                           # [B,T,T_ref]
        w          = F.softmax(-dist_sq / tau, dim=-1)            # [B,T,T_ref]

        # ---------- 3. 绝对横向误差 ----------
        lat_proj = (delta * nor.unsqueeze(1)).sum(-1)             # [B,T,T_ref]
        lat_sq = (w * lat_proj.pow(2)).sum(-1)                    # [B, T]
        lat_abs = torch.sqrt(lat_sq + 1e-8)                       # [B, T]
        # 使用 torch.where 保证梯度可导（在 x=1 处一阶导连续）
        mask_small = lat_abs < 1.0
        huber_val = torch.where(
            mask_small,
            0.5 * lat_abs.pow(2),   # 当 lat_abs < 1
            lat_abs - 0.5           # 当 lat_abs >= 1
        )  # [B, T]
        # Huber: β = 1
        huber = torch.where(lat_abs < 1.0,
                            0.5 * lat_abs.pow(2),
                            lat_abs - 0.5)                        # [B,T]

        # ---------- 4. mask & reduce ----------
        valid_mask_f = valid_mask.float()
        huber = huber * valid_mask_f                              # [B,T]

        # （可选）在这里做归一化，便于不同 batch 对齐梯度规模
        eps = 1e-6  # 防止全无效时除 0
        huber = huber.sum() / (valid_mask_f.sum() + eps)
        
        return huber                                              # 

    def _batch_angle_smoothness_cost(self, points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        计算批量路径的角度平滑代价，忽略不同段长度影响，仅惩罚方向变化。

        参数:
            points: torch.Tensor, shape = (B, T, 2)
                B: batch size
                T: 每条轨迹的点数
                2: x,y 坐标
            eps: float
                防止除以零的极小正数，默认 1e-6。

        返回:
            cost: torch.Tensor, shape = (B, 1)
                对每条轨迹，计算 (1/(T-2)) * sum_{i=0..T-3} ||u_{i+1} - u_i||^2，
                其中 u_i = (p_{i+1} - p_i) / (||p_{i+1}-p_i|| + eps)。
                如果 T < 3，则整条轨迹的 cost 设为 0。
        """
        B, T, _ = points.shape
        device = points.device
        dtype = points.dtype

        if T < 2:
            # 如果 T<2，则根本没有“段”，输出全 0
            return torch.zeros((B, T, 1), device=points.device, dtype=points.dtype)
        if T < 3:
            # 如果 T=2，则只有一个方向向量，无法计算相邻两段之间的角度变化
            # 所以所有点都设为 0
            return torch.zeros((B, T, 1), device=points.device, dtype=points.dtype)

        # 1) 计算每一段向量 v_i = p[:, i+1, :] - p[:, i, :]
        #    结果 shape = (B, T-1, 2)
        v = points[:, 1:, :] - points[:, :-1, :]

        # 2) 计算每段向量的长度 norm_v = ||v_i||，shape = (B, T-1, 1)
        norm_v = torch.norm(v, dim=2, keepdim=True)
        u = v / (norm_v + eps)
        # 3) 计算相邻两段单位向量 (u_i, u_{i+1}) 之间的“有向夹角” θ_i
        ux = u[..., 0]  # [B, T-1]
        uy = u[..., 1]  # [B, T-1]

        #    再取 u_shifted = u 后移一位，对应 u_{i+1}
        ux_next = ux[:, 1:]  # [B, T-2]
        uy_next = uy[:, 1:]  # [B, T-2]
        ux_curr = ux[:, :-1] # [B, T-2]
        uy_curr = uy[:, :-1] # [B, T-2]

        #    dot = u_curr · u_next
        dot = ux_curr * ux_next + uy_curr * uy_next           # [B, T-2]
        #    cross = ux_curr * uy_next - uy_curr * ux_next
        cross = ux_curr * uy_next - uy_curr * ux_next         # [B, T-2]

        #    θ = atan2(cross, dot), shape → (B, T-2)
        theta = torch.atan2(cross, dot)                        # [B, T-2]
        angle_cost = theta.pow(2)                              # [B, T-2]
        # 4) 把 angle_cost 填到 cost 张量中，cost[b, i+1] = (θ_i)^2
        #    cost 的形状初始化为 (B, T)
        cost = torch.zeros((B, T), device=points.device, dtype=points.dtype)  # [B, T]
        #    i = 0..T-3 对应的 θ_i 填到 cost[:, i+1]
        cost[:, 1:T-1] = angle_cost  # 把 [B, T-2] 放到 cost 的 [1..T-2] 位置

        # 5) 最后 unsqueeze(-1) 得到 (B, T, 1) 返回
        return cost.unsqueeze(-1)  # [B, T, 1]

    def _compute_metrics(self, res, data, prefix) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        
        metrics = {}
        return metrics

    def _log_step(
        self,
        loss,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        bs = None
        if "reg_loss" in objectives:
            for key, val in objectives.items():
                if isinstance(val, torch.Tensor) and val.ndim > 0:
                    bs = val.shape[0]
                    break
        if bs is None:
            bs = 1
            
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True if prefix == "train" else False,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                prog_bar=True if prefix == "train" else False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix in ["val", "train"]),
                on_step=False,
                on_epoch=True,
                batch_size=int(bs),
                sync_dist=True,
            )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        # print("features:",features)
        # print(f"LightingTrainer forward forward forward forward forward")

        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # assert len(inter_params) == 0
        # assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer=optimizer,
            epoch=self.epochs,
            warm_up_epoch=self.warmup_epochs,
        )
        return [optimizer], [scheduler]

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     sh_layer = self.model.encoder_blocks[-1].mlp.fc2
    #     norms = grad_norm(sh_layer, norm_type=2)
    #     self.log_dict(norms)

    def mgda_find_scaler(self, losses, skip=5):
        if self.global_step%skip!=0:
            return torch.stack([l*w for l,w in zip(losses, self.weights)]).sum()
        sh_layer = self.model.encoder_blocks[-1].mlp.fc2
        gw = []
        for i in range(len(losses)):
            dl = torch.autograd.grad(losses[i], sh_layer.parameters(), retain_graph=True, create_graph=True, allow_unused=True)[0]
            # dl = torch.norm(dl)
            gw.append([dl])
        # sol, min_norm = MinNormSolver.find_min_norm_element(gw)
        sol, min_norm = None,None
        self.weights = sol
        weighted_loss = torch.stack([l*w for l,w in zip(losses, sol)]).sum()
        return weighted_loss