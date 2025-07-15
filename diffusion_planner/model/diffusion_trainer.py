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
from diffusion_planner.metrics import MR, minADE, minFDE, mulADE, MVNLoss
from diffusion_planner.metrics.prediction_avg_ade import PredAvgADE
from diffusion_planner.metrics.prediction_avg_fde import PredAvgFDE
from diffusion_planner.optim.warmup_cos_lr import WarmupCosLR

from .loss.esdf_collision_loss import ESDFCollisionLoss
from pytorch_lightning.utilities import grad_norm
from diffusion_planner.utils.min_norm_solvers import MinNormSolver

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
        use_collision_loss=True,
        use_contrast_loss=False,
        regulate_yaw=False,
        objective_aggregate_mode: str = "mean",
        mul_ade_loss: list[str] = ['phase_loss', 'scale_loss'],
        dynamic_weight: bool = True,
        max_horizon: int = 10,
        use_dwt: bool = False,
        learning_output: str = 'velocity',
        init_weights: list[float] = [1, 1, 1, 1, 1, 1, 1, 1],
        wavelet: list[str] = ['cgau1', 'constant', 'bior1.3', 'constant'],
        wtd_with_history: bool = False,
        approximation_norm: bool = False,
        time_decay: bool = False,
        time_norm: bool = False,
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.objective_aggregate_mode = objective_aggregate_mode
        self.history_steps = model.history_steps
        self.use_collision_loss = use_collision_loss
        self.use_contrast_loss = use_contrast_loss
        self.regulate_yaw = regulate_yaw

        self.radius = model.radius
        self.num_modes = model.num_modes
        self.mode_interval = self.radius / self.num_modes
        self.time_decay = time_decay
        self.time_norm = time_norm

        if use_collision_loss:
            self.collision_loss = ESDFCollisionLoss()
        self.mul_ade = mulADE(k=1, 
                              with_grad=True,
                              mul_ade_loss=mul_ade_loss, 
                              max_horizon=max_horizon, 
                              learning_output=learning_output,
                              wtd_with_history=wtd_with_history,
                              wavelet=wavelet,
                              approximation_norm=approximation_norm,
                              use_dwt=use_dwt,
                              ).to(self.device)
        self.dynamic_weight = dynamic_weight
        print(f"self.device: {self.device}")
        init_weights = [float(w) for w in init_weights]
        self.weights = torch.tensor(init_weights, dtype=torch.float32)
        self.weights = self.weights.to(self.device)
        print(f"self.weights dtype after to device: {self.weights.dtype}")
        if self.dynamic_weight:
            # self.weights = torch.autograd.Variable(self.weights, requires_grad=True)
            self.weights.requires_grad = True
        self.mvn_loss = MVNLoss(k=3, with_grad=True).to(self.device)
        print('WARNING: Overall future time horizon is set to 80')
        self.OT = 80

    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            [
                minADE().to(self.device),
                minFDE().to(self.device),
                MR(miss_threshold=2).to(self.device),
                PredAvgADE().to(self.device),
                PredAvgFDE().to(self.device),
            ]
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """

        features, targets, scenarios = batch
        
        # print(f"features:{type(features)}")
        # print(f"targets:type:{type(targets)}")
        # print(f"scenarios:type:{type(scenarios)}")

        # for key, val in features.items(): print(f"features_{key}:type:{type(val)}")
        # for key, val in targets.items(): print(f"targets_{key}:type:{type(val)}")
        # for item in scenarios: print(f"scenarios type:{type(item)}")
        
        # features 中只有一个 feature 作为键的数据，值的类型为该工程定义的class diffusion_planner.features.pluto_feature.DiffusionFeature
        # targets 中只有一个 trajectory 作为键的数据，值的类型为nuplan.planning.training.preprocessing.features.trajectory.Trajectory
        # scenarios 为list，中数据类型为 nuplan.planning.scenario_builder.cache.cached_scenario.CachedScenario
        # print(f"features['feature'].data.type: {type(features['feature'].data)}")
        # for key,val in features['feature'].data.items(): print(f"features['feature'].data.key {key}: {type(val)}")
        '''
        features['feature'].data.key agent: <class 'dict'>
        features['feature'].data.key map: <class 'dict'>
        features['feature'].data.key reference_line: <class 'dict'>
        features['feature'].data.key static_objects: <class 'dict'>
        features['feature'].data.key data_n_valid_mask: <class 'torch.Tensor'>
        features['feature'].data.key data_n_type: <class 'torch.Tensor'>
        features['feature'].data.key current_state: <class 'torch.Tensor'>
        features['feature'].data.key origin: <class 'torch.Tensor'>
        features['feature'].data.key angle: <class 'torch.Tensor'>
        features['feature'].data.key cost_maps: <class 'torch.Tensor'>
        '''
        # for key,val in features['feature'].data["agent"].items(): print(f"agent {key}: {val.shape}")
        """
        agent position: torch.Size([12, 49, 101, 2]) # 12个batch，49个agent，101个时间步，2个坐标
        agent heading: torch.Size([12, 49, 101])
        agent velocity: torch.Size([12, 49, 101, 2])
        agent shape: torch.Size([12, 49, 101, 2])
        agent category: torch.Size([12, 49])
        agent valid_mask: torch.Size([12, 49, 101])
        agent target: torch.Size([12, 49, 80, 3]) # 就是最后80个时间步的目标位置与朝向
        """
        # print(f"features['feature'].data['agent']['position'][0][0][21:]={features['feature'].data['agent']['position'][0][0][21:31]}")
        # print(f"features['feature'].data['agent']['heading'][0][0][21:]={features['feature'].data['agent']['heading'][0][0][21:31]}")
        # print(f"features['feature'].data['agent']['target'][0][0][21:]={features['feature'].data['agent']['target'][0][0][:10]}")
        # for key,val in features['feature'].data["map"].items(): print(f"map {key}: {val.shape}")
        '''
        map point_position: torch.Size([12, 144, 3, 20, 2])
        map point_vector: torch.Size([12, 144, 3, 20, 2])
        map point_orientation: torch.Size([12, 144, 3, 20])
        map point_side: torch.Size([12, 144, 3])
        map polygon_center: torch.Size([12, 144, 3])
        map polygon_position: torch.Size([12, 144, 2])
        map polygon_orientation: torch.Size([12, 144])
        map polygon_type: torch.Size([12, 144])
        map polygon_on_route: torch.Size([12, 144])
        map polygon_tl_status: torch.Size([12, 144])
        map polygon_has_speed_limit: torch.Size([12, 144])
        map polygon_speed_limit: torch.Size([12, 144])
        map polygon_road_block_id: torch.Size([12, 144])
        map valid_mask: torch.Size([12, 144, 20])
        '''
        # print(f"targets['trajectory'].data.shape:{targets['trajectory'].data.shape}")
        # print(f"features['feature'].data['agent']['position'][0][0][21::10]={features['feature'].data['agent']['position'][0][0][21::10]}")
        # print(f"targets['trajectory'].data[0]:{targets['trajectory'].data[0]}")
        
        # print(f"---------------------")

        res = self.forward(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        # 合并 losses（去掉总 loss）进 metrics
        # extra_metrics = {
        #     f"{prefix}/{k}": v.detach().float() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
        #     for k, v in losses.items() if k != "loss"
        # }
        # metrics.update(extra_metrics)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"] if self.training else 0.0

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        bs, _, T, _ = res["prediction"].shape

        if self.use_contrast_loss:
            train_num = (bs // 3) * 2 if self.training else bs
        else:
            train_num = bs

        trajectory, probability, prediction = (
            res["trajectory"][:train_num],
            res["probability"][:train_num],
            res["prediction"][:train_num],
        )
        ref_free_trajectory = res.get("ref_free_trajectory", None)
        end = -self.OT+T if T < self.OT else None
        targets_pos = data["agent"]["target"][:train_num, :, -self.OT:end]
        valid_mask = data["agent"]["valid_mask"][:train_num, :, -self.OT:end]
        targets_vel = data["agent"]["velocity"][:train_num, :, -self.OT:end]
        target = torch.cat(
            [
                targets_pos[..., :2],
                torch.stack(
                    [targets_pos[..., 2].cos(), targets_pos[..., 2].sin()], dim=-1
                ),
                targets_vel,
            ],
            dim=-1,
        )

        # planning loss
        ego_reg_loss, ego_cls_loss, collision_loss, lane_deviation_loss, smoothness_loss = self.get_planning_loss(
            data, trajectory, probability, valid_mask[:, 0], target[:, 0], train_num
        )
        if ref_free_trajectory is not None:
            ego_ref_free_reg_loss = F.smooth_l1_loss(
                ref_free_trajectory[:train_num],
                target[:, 0, :, : ref_free_trajectory.shape[-1]],
                reduction="none",
            ).sum(-1)
            ego_ref_free_reg_loss = (
                ego_ref_free_reg_loss * valid_mask[:, 0]
            ).sum() / valid_mask[:, 0].sum()
        else:
            ego_ref_free_reg_loss = ego_reg_loss.new_zeros(1)

        # prediction loss
        prediction_loss = self.get_prediction_loss(
            data, prediction, valid_mask[:, 1:], target[:, 1:]
        ) if 'mvn' not in res.keys() else self.mvn_loss(res, data)

        if self.training and self.use_contrast_loss:
            contrastive_loss = self._compute_contrastive_loss(
                res["hidden"], data["data_n_valid_mask"]
            )
        else:
            contrastive_loss = prediction_loss.new_zeros(1)

        scope_loss = self.mul_ade(res, data)
        
        loss = (
            ego_reg_loss*self.weights[0] # 回归损失（路径近似程度）
            + ego_cls_loss*self.weights[1] # 分类损失（体现决策近似程度）
            + prediction_loss*self.weights[2] #运动预测损失
            + contrastive_loss # 三元组对比损失,为了学习一个良好的嵌入空间，使得来自同一类别或同一正样本对的向量更相似，而来自不同类别或负样本对的向量更不相似
            + collision_loss*self.weights[3] # 碰撞风险损失
            + ego_ref_free_reg_loss*self.weights[4] # 无参考回归损失，使用数据集GT作为参考Target
            + scope_loss*self.weights[5] # 路径细节损失
            # + lane_deviation_loss*self.weights[6] # 路径偏离损失
            + smoothness_loss*self.weights[7] # 路径平滑损失
        )
        if self.training and self.dynamic_weight:
            self.losses = [ego_reg_loss, 
                           ego_cls_loss, 
                           prediction_loss, 
                            # contrastive_loss[0], 
                            collision_loss, 
                            ego_ref_free_reg_loss, 
                            scope_loss,
                            # lane_deviation_loss,
                            smoothness_loss
                            ]
            loss = self.mgda_find_scaler(self.losses)

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.item(),
            "cls_loss": ego_cls_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "collision_loss": collision_loss.item(),
            "ref_free_reg_loss": ego_ref_free_reg_loss.item(),
            "scope_loss": scope_loss.item(),
            # "lane_deviation_loss": lane_deviation_loss.item(),
            "smoothness_loss": smoothness_loss.item(),
            "alpha_reg_loss": self.weights[0],
            "alpha_cls_loss": self.weights[1],
            "alpha_prediction_loss": self.weights[2],
            # "alpha_contrastive_loss": self.weights[5],
            "alpha_collision_loss": self.weights[3],
            "alpha_ref_free_reg_loss": self.weights[4],
            "alpha_scope_loss": self.weights[5],
            # "alpha_ref_dev_loss": self.weights[6],
            "alpha_smooth_loss": self.weights[7]
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

    def _compute_contrastive_loss(
        self, hidden, valid_mask, normalize=True, tempreture=0.1
    ):
        """
        Compute triplet loss

        Args:
            hidden: (3*bs, D)
        """
        if normalize:
            hidden = F.normalize(hidden, dim=1, p=2)

        if not valid_mask.any():
            return hidden.new_zeros(1)

        x_a, x_p, x_n = hidden.chunk(3, dim=0)

        x_a = x_a[valid_mask]
        x_p = x_p[valid_mask]
        x_n = x_n[valid_mask]

        logits_ap = (x_a * x_p).sum(dim=1) / tempreture
        logits_an = (x_a * x_n).sum(dim=1) / tempreture
        labels = x_a.new_zeros(x_a.size(0)).long()

        triplet_contrastive_loss = F.cross_entropy(
            torch.stack([logits_ap, logits_an], dim=1), labels
        )
        return triplet_contrastive_loss

    def _compute_metrics(self, res, data, prefix) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        # get top 6 modes
        trajectory, probability = res["trajectory"], res["probability"]
        r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        bs, R, M, T, _ = trajectory.shape
        trajectory = trajectory.reshape(bs, R * M, T, -1)
        probability = probability.reshape(bs, R * M)
        top_k_prob, top_k_index = probability.topk(6, dim=-1)
        top_k_traj = trajectory[torch.arange(bs)[:, None], top_k_index]

        outputs = {
            "trajectory": top_k_traj[..., :2],
            "probability": top_k_prob,
            "prediction": res["prediction"][..., :2],
            "prediction_target": data["agent"]["target"][:, 1:],
            "valid_mask": data["agent"]["valid_mask"][:, 1:, self.history_steps :],
        }
        target = data["agent"]["target"][:, 0]

        metrics = self.metrics[prefix](outputs, target)
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

        data = [
            features['agent']['position'],  #0
            features['agent']['heading'],#1
            features['agent']['velocity'],#2
            features['agent']['shape'],#3
            features['agent']['category'],#4
            features['agent']['valid_mask'],#5

            features['map']['point_position'],#6
            features['map']['point_vector'],#7
            features['map']['point_orientation'],#8
            features['map']['polygon_center'],#9
            features['map']['polygon_type'],#10
            features['map']['polygon_on_route'],#11
            features['map']['polygon_tl_status'],#12
            features['map']['polygon_has_speed_limit'],#13
            features['map']['polygon_speed_limit'],#14
            features['map']['valid_mask'],#15

            features['reference_line']['position'],#16
            features['reference_line']['vector'],#17
            features['reference_line']['orientation'],#18
            features['reference_line']['valid_mask'],#19

            features['static_objects']['position'],#20
            features['static_objects']['heading'],#21
            features['static_objects']['shape'],#22
            features['static_objects']['category'],#23
            features['static_objects']['valid_mask'],#24

            features['current_state'],#25
        ]
        return self.model(data)
        # return self.model(features)

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
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

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
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
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
        sol, min_norm = MinNormSolver.find_min_norm_element(gw)
        self.weights = sol
        weighted_loss = torch.stack([l*w for l,w in zip(losses, sol)]).sum()
        return weighted_loss