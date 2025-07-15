export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

export NUPLAN_DATA_ROOT="/home/bydguikong/nuplan/data"
export NUPLAN_MAPS_ROOT="/home/bydguikong/nuplan/maps"
export WS="/home/bydguikong/nuplan"
export NUPLAN_EXP_ROOT="$WS/exp"
export HYDRA_FULL_ERROR=1
# export CKPT="/home/bydguikong/yy_ws/PlanScope/checkpoints/last-local-250606-01.ckpt"

CFG_DIR="./diffusion_planner/config"

echo "====Start Sanity Check====" &&
CUDA_VISIBLE_DEVICES=0  python run_training.py \
  --config-path "$CFG_DIR" \
  py_func=train +training=train_diffusion \
  worker=single_machine_thread_pool worker.max_workers=4 \
  epochs=2 warmup_epochs=1 \
  scenario_builder=nuplan cache.cache_path=$WS/exp/sanity_check cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=4 data_loader.params.num_workers=1 \
  &&
  
echo "====Start training(hear)====" &&
CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=train +training=train_diffusion \
  worker=single_machine_thread_pool worker.max_workers=1 \
  scenario_builder=nuplan_mini \
  cache.use_cache_without_dataset=true \
  cache.cache_path=$WS/exp/val14_benchmark \
  scenario_filter=val14_benchmark1 \
  data_loader.params.batch_size=1 data_loader.params.num_workers=24 \
  lightning.trainer.params.val_check_interval=12 \
  lightning.trainer.params.strategy=ddp_find_unused_parameters_true \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  data_loader.datamodule.train_fraction=1.0 \
  data_loader.datamodule.val_fraction=1.0 \
  data_loader.datamodule.test_fraction=1.0 \
  &&
  echo "====Training End===="
  
