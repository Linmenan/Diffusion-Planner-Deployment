from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from torch.nn.utils.rnn import pad_sequence

from diffusion_planner.utils.utils import to_device, to_numpy, to_tensor


@dataclass
class DiffusionFeature(AbstractModelFeature):
    data: Dict[str, Any]  # anchor sample
    # data_p: Dict[str, Any] = None  # positive sample
    # data_n: Dict[str, Any] = None  # negative sample
    # data_n_info: Dict[str, Any] = None  # negative sample info

    @classmethod
    def collate(cls, feature_list: List[DiffusionFeature]) -> DiffusionFeature:
        # 如果输入的列表为空，则返回空
        if not feature_list:
            return cls(data={})

        batch_data = {}
        first_data = feature_list[0].data

        # 1. 定义不同处理方式的键
        # 这些键对应的数据是固定尺寸的，将被堆叠
        stack_keys = ["ego_current_state", "ego_agent_future"]
        # 这些键是字符串或元数据，将被收集到列表中
        # list_keys = ["map_name", "token"]
        list_keys = []
        # 所有其他的键默认被认为是可变长度的，需要填充
        # 这包括 'neighbor_*', 'static_objects', 以及所有地图特征（如 'lanes'）
        pad_keys = [k for k in first_data.keys() if k not in stack_keys and k not in list_keys]

        # 辅助函数，确保输入是 torch.Tensor
        def to_tensor_list(items: List[Any]) -> List[torch.Tensor]:
            return [
                torch.from_numpy(item) if isinstance(item, np.ndarray) else item
                for item in items
            ]

        

        for key in pad_keys:
            items_to_pad = to_tensor_list([f.data[key] for f in feature_list])
            batch_data[key] = pad_sequence(items_to_pad, batch_first=True)
        for key in stack_keys:
            items_to_stack = to_tensor_list([f.data[key] for f in feature_list])
            batch_data[key] = torch.stack(items_to_stack, dim=0)

        # 3. 单独处理元数据键（这些数据不参与多样本组合）
        for key in list_keys:
            batch_data[key] = [f.data[key] for f in feature_list]

        return cls(data=batch_data)

    def to_feature_tensor(self) -> DiffusionFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_tensor(v)

        # if self.data_p is not None:
        #     new_data_p = {}
        #     for k, v in self.data_p.items():
        #         new_data_p[k] = to_tensor(v)
        # else:
        #     new_data_p = None

        # if self.data_n is not None:
        #     new_data_n = {}
        #     new_data_n_info = {}
        #     for k, v in self.data_n.items():
        #         new_data_n[k] = to_tensor(v)
        #     for k, v in self.data_n_info.items():
        #         new_data_n_info[k] = to_tensor(v)
        # else:
        #     new_data_n = None
        #     new_data_n_info = None

        return DiffusionFeature(
            data=new_data,
            # data_p=new_data_p,
            # data_n=new_data_n,
            # data_n_info=new_data_n_info,
        )

    def to_numpy(self) -> DiffusionFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_numpy(v)
        # if self.data_p is not None:
        #     new_data_p = {}
        #     for k, v in self.data_p.items():
        #         new_data_p[k] = to_numpy(v)
        # else:
        #     new_data_p = None
        # if self.data_n is not None:
        #     new_data_n = {}
        #     for k, v in self.data_n.items():
        #         new_data_n[k] = to_numpy(v)
        # else:
        #     new_data_n = None
        return DiffusionFeature(
            data=new_data, 
            # data_p=new_data_p, 
            # data_n=new_data_n
            )

    def to_device(self, device: torch.device) -> DiffusionFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_device(v, device)
        return DiffusionFeature(data=new_data)

    def serialize(self) -> Dict[str, Any]:
        return {"data": self.data}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> DiffusionFeature:
        return DiffusionFeature(data=data["data"])

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError

    @property
    def is_valid(self) -> bool:
        
        return True

    @classmethod
    def normalize(
        self, data, first_time=False, radius=None, hist_steps=21
    ) -> DiffusionFeature:
        

        return DiffusionFeature(data=data)
