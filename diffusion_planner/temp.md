```
diffusion_planner
├── __pycache__
│   └── __init__.cpython-39.pyc
├── config
│   ├── __pycache__
│   │   └── __init__.cpython-39.pyc
│   ├── custom_trainer
│   │   └── diffusion_trainer.yaml
│   ├── data_augmentation
│   │   └── diffusion_scenario_generator.yaml
│   ├── lightning
│   │   └── custom_lightning.yaml
│   ├── model
│   │   └── diffusion_planner.yaml
│   ├── planner
│   │   ├── __init__.py
│   │   ├── diffusion_planner.yaml
│   │   └── diffusion_planner_guidance.yaml
│   ├── scenario_filter
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-39.pyc
│   │   ├── __init__.py
│   │   ├── test14-hard.yaml
│   │   ├── test14-random.yaml
│   │   ├── test_one_scenatio.yaml
│   │   ├── training_scenarios_tiny.yaml
│   │   ├── val14-collision.yaml
│   │   ├── val14.yaml
│   │   └── yy.yaml
│   ├── training
│   │   └── train_diffusion.yaml
│   ├── __init__.py
│   ├── default_simulation.yaml
│   └── default_training.yaml
├── custom_training
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc
│   │   ├── custom_datamodule.cpython-39.pyc
│   │   └── custom_training_builder.cpython-39.pyc
│   ├── __init__.py
│   ├── custom_datamodule.py
│   └── custom_training_builder.py
├── data_augmentation
│   ├── __pycache__
│   │   └── scope_scenario_generator.cpython-39.pyc
│   ├── contrastive_scenario_generator.py
│   └── diffusion_scenario_generator.py
├── data_process
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc
│   │   ├── agent_process.cpython-39.pyc
│   │   ├── data_processor.cpython-39.pyc
│   │   ├── ego_process.cpython-39.pyc
│   │   ├── map_process.cpython-39.pyc
│   │   ├── roadblock_utils.cpython-39.pyc
│   │   └── utils.cpython-39.pyc
│   ├── __init__.py
│   ├── agent_process.py
│   ├── data_processor.py
│   ├── ego_process.py
│   ├── map_process.py
│   ├── roadblock_utils.py
│   └── utils.py
├── feature_builders
│   ├── __pycache__
│   │   ├── common.cpython-39.pyc
│   │   ├── diffusion_feature_builder.cpython-39.pyc
│   │   ├── nuplan_scenario_render.cpython-39.pyc
│   │   ├── pluto_feature_builder.cpython-39.pyc
│   │   └── scope_feature_builder.cpython-39.pyc
│   ├── common.py
│   ├── diffusion_feature_builder.py
│   └── nuplan_scenario_render.py
├── features
│   ├── __pycache__
│   │   ├── diffusion_feature.cpython-39.pyc
│   │   ├── pluto_feature.cpython-39.pyc
│   │   └── scope_feature.cpython-39.pyc
│   └── diffusion_feature.py
├── model
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc
│   │   └── diffusion_planner.cpython-39.pyc
│   ├── diffusion_utils
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-39.pyc
│   │   │   ├── dpm_solver_pytorch.cpython-39.pyc
│   │   │   ├── sampling.cpython-39.pyc
│   │   │   └── sde.cpython-39.pyc
│   │   ├── __init__.py
│   │   ├── dpm_solver_pytorch.py
│   │   ├── sampling.py
│   │   └── sde.py
│   ├── guidance
│   │   ├── collision.py
│   │   ├── documentation_guidance.md
│   │   └── guidance_wrapper.py
│   ├── module
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-39.pyc
│   │   │   ├── custom_multihead_attention.cpython-39.pyc
│   │   │   ├── decoder.cpython-39.pyc
│   │   │   ├── dit.cpython-39.pyc
│   │   │   ├── encoder.cpython-39.pyc
│   │   │   └── mixer.cpython-39.pyc
│   │   ├── __init__.py
│   │   ├── custom_multihead_attention.py
│   │   ├── decoder.py
│   │   ├── dit.py
│   │   ├── encoder.py
│   │   └── mixer.py
│   ├── __init__.py
│   ├── diffusion_planner.py
│   └── diffusion_trainer.py
├── planner
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc
│   │   └── planner.cpython-39.pyc
│   ├── __init__.py
│   └── planner.py
├── scenario_manager
│   ├── __pycache__
│   │   ├── cost_map_manager.cpython-39.pyc
│   │   ├── occupancy_map.cpython-39.pyc
│   │   ├── route_manager.cpython-39.pyc
│   │   └── scenario_manager.cpython-39.pyc
│   ├── utils
│   │   ├── __pycache__
│   │   │   ├── bfs_roadblock.cpython-39.pyc
│   │   │   ├── dijkstra.cpython-39.pyc
│   │   │   └── route_utils.cpython-39.pyc
│   │   ├── bfs_roadblock.py
│   │   ├── dijkstra.py
│   │   └── route_utils.py
│   ├── cost_map_manager.py
│   ├── occupancy_map.py
│   ├── route_manager.py
│   └── scenario_manager.py
├── utils
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc
│   │   ├── config.cpython-39.pyc
│   │   ├── normalizer.cpython-39.pyc
│   │   ├── train_utils.cpython-39.pyc
│   │   ├── utils.cpython-39.pyc
│   │   └── vis.cpython-39.pyc
│   ├── __init__.py
│   ├── config.py
│   ├── data_augmentation.py
│   ├── dataset.py
│   ├── ddp.py
│   ├── lr_schedule.py
│   ├── normalizer.py
│   ├── tb_log.py
│   ├── train_utils.py
│   ├── utils.py
│   └── vis.py
├── __init__.py
├── loss.py
├── temp.md
└── train_epoch.py

```