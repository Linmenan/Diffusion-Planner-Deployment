###################################
# User Configuration Section
###################################
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:"/home/bydguikong/yy_ws/nuplan-devkit/"
export NUPLAN_DATA_ROOT="/home/bydguikong/nuplan/dataset"
export NUPLAN_MAPS_DB="/home/bydguikong/nuplan/dataset/maps"
export NUPLAN_DB_FILES="/home/bydguikong/nuplan/dataset/nuplan-v1.1/splits/mini"

NUPLAN_DATA_PATH="/home/bydguikong/nuplan/dataset" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/home/bydguikong/nuplan/dataset/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")
TRAIN_SET_PATH="/home/bydguikong/yy_ws/Diffusion-Planner/detect_data"
###################################
echo "--- Environment and Paths ---"
echo "NUPLAN_DATA_ROOT is set to: $NUPLAN_DATA_ROOT"
echo "NUPLAN_MAPS_DB is set to: $NUPLAN_MAPS_DB"
echo "NUPLAN_DB_FILES is set to: $NUPLAN_DB_FILES"
echo "-----------------------------"
echo "Starting python script..."

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \

