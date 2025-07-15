import numpy as np

def extract_model_inputs(raw_inputs):
    for k,v in raw_inputs.items():
        print(f"raw_inputs.{k}:{v.shape}")
    # 1. neighbor_agents_past
    # 假设你原始有 agent.position 和 agent.velocity 形状类似 (1,7,21,2)
    # 先转置成 (1, 21, 7, 2)
    agent_pos = np.transpose(raw_inputs['agent.position'], (0, 2, 1, 3))  # (1,21,7,2)
    agent_vel = np.transpose(raw_inputs['agent.velocity'], (0, 2, 1, 3))  # (1,21,7,2)

    # 你要的是 (1, 32, 21, 11)
    # 先合并 position 和 velocity 的最后维度（2+2=4），拼成特征维
    # 先 concat 最后维度
    neighbor_agents = np.concatenate([agent_pos, agent_vel], axis=-1)  # (1,21,7,4)
    # 转换维度到 (1, 32, 21, 11)，这里只能pad或者截断
    # 先转成 (1, 32, 21, 11)的空数组
    new_shape = (1, 32, 21, 11)
    neighbor_agents_past = np.zeros(new_shape, dtype=neighbor_agents.dtype)
    # 只copy最大可用部分
    copy_dim1 = min(neighbor_agents.shape[1], new_shape[1])  # 21
    copy_dim2 = min(neighbor_agents.shape[2], new_shape[2])  # 7 vs 21 ?
    copy_dim3 = min(neighbor_agents.shape[3], new_shape[3])  # 4 vs 11
    # 由于第2维7 < 21, 只能填7条，然后第二维填7而不是21
    neighbor_agents_past[:, :copy_dim1, :copy_dim2, :copy_dim3] = neighbor_agents[:, :copy_dim1, :copy_dim2, :copy_dim3]

    # 2. ego_current_state: 原 (1,7) 或 (7,)
    ego_current_state_raw = raw_inputs['current_state']
    # 确保是 (1,10)
    if ego_current_state_raw.ndim == 1:
        ego_current_state_raw = ego_current_state_raw[np.newaxis, :]
    ego_current_state = np.zeros((1, 10), dtype=ego_current_state_raw.dtype)
    copy_len = min(ego_current_state_raw.shape[1], 10)
    ego_current_state[:, :copy_len] = ego_current_state_raw[:, :copy_len]

    # 3. static_objects: 原(1,3,4) 目标(1,5,10)
    pos = raw_inputs['static_objects.position']      # (1,3,2)
    heading = raw_inputs['static_objects.heading']   # (1,3)
    shape = raw_inputs['static_objects.shape']       # (1,3,2)
    category = raw_inputs['static_objects.category'] # (1,3)

    heading = heading[..., np.newaxis]                # (1,3,1)
    category = category[..., np.newaxis]              # (1,3,1)

    static_objects_raw = np.concatenate([pos, heading, shape, category], axis=2)  # (1,3,6)
    # 要求是10维，pad到10
    static_objects = np.zeros((1, 5, 10), dtype=static_objects_raw.dtype)
    copy_n = min(3, 5)
    copy_f = min(6, 10)
    static_objects[:, :copy_n, :copy_f] = static_objects_raw[:, :copy_n, :copy_f]

    # 4. lanes: 原(40,20,2) 目标(1,70,20,12)
    lanes_raw = raw_inputs['map.point_position'][0]  # (40,3,20,2)
    # 取某一维简化成 (40,20,2)
    lanes_raw = lanes_raw[:,0,:,:]  # (40,20,2)

    lanes = np.zeros((1, 70, 20, 12), dtype=lanes_raw.dtype)
    copy_n = min(40, 70)
    copy_m = min(20, 20)
    copy_f = min(2, 12)
    lanes[0, :copy_n, :copy_m, :copy_f] = lanes_raw[:copy_n, :copy_m, :copy_f]

    # 5. lanes_speed_limit (40,) -> (1,70,1)
    lanes_speed_limit_raw = raw_inputs['map.polygon_speed_limit'][0]  # (40,)
    lanes_speed_limit = np.zeros((1, 70, 1), dtype=lanes_speed_limit_raw.dtype)
    lanes_speed_limit[0, :40, 0] = lanes_speed_limit_raw

    # 6. lanes_has_speed_limit (40,) -> (1,70,1)
    lanes_has_speed_limit_raw = raw_inputs['map.polygon_has_speed_limit'][0]  # (40,)
    lanes_has_speed_limit = np.zeros((1, 70, 1), dtype=lanes_has_speed_limit_raw.dtype)
    lanes_has_speed_limit[0, :40, 0] = lanes_has_speed_limit_raw

    # 7. route_lanes (7,20,2) -> (1,25,20,12)
    route_lanes_raw = raw_inputs['map.point_position'][0, :7, 0, :, :]  # (7,20,2)
    route_lanes = np.zeros((1, 25, 20, 12), dtype=route_lanes_raw.dtype)
    copy_n = min(7, 25)
    copy_m = min(20, 20)
    copy_f = min(2, 12)
    route_lanes[0, :copy_n, :copy_m, :copy_f] = route_lanes_raw[:copy_n, :copy_m, :copy_f]

    # 8. route_lanes_speed_limit (7,) -> (1,25,1)
    route_lanes_speed_limit_raw = raw_inputs['map.polygon_speed_limit'][0, :7]
    route_lanes_speed_limit = np.zeros((1, 25, 1), dtype=route_lanes_speed_limit_raw.dtype)
    route_lanes_speed_limit[0, :7, 0] = route_lanes_speed_limit_raw

    # 9. route_lanes_has_speed_limit (7,) -> (1,25,1)
    route_lanes_has_speed_limit_raw = raw_inputs['map.polygon_has_speed_limit'][0, :7]
    route_lanes_has_speed_limit = np.zeros((1, 25, 1), dtype=route_lanes_has_speed_limit_raw.dtype)
    route_lanes_has_speed_limit[0, :7, 0] = route_lanes_has_speed_limit_raw

    return {
        "neighbor_agents_past": neighbor_agents_past.astype(np.float32),
        "ego_current_state": ego_current_state.astype(np.float32),
        "static_objects": static_objects.astype(np.float32),
        "lanes": lanes.astype(np.float32),
        "lanes_speed_limit": lanes_speed_limit.astype(np.float32),
        "lanes_has_speed_limit": lanes_has_speed_limit.astype(bool),
        "route_lanes": route_lanes.astype(np.float32),
        # "route_lanes_speed_limit": route_lanes_speed_limit.astype(np.float32),
        # "route_lanes_has_speed_limit": route_lanes_has_speed_limit.astype(bool),
    }
