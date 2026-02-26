conda create -n uniad python=3.8 -y
conda activate uniad
conda install cudatoolkit=11.1.1 -c conda-forge
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
export PATH=YOUR_GCC_PATH/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8/
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
cd mmdetection3d
pip install scipy==1.7.3
pip install scikit-image==0.20.0
pip install -v -e .
cd ..
pip install -r requirements.txt
pip install -U "yapf==0.32.0"
pip install -U "numpy==1.21.6"
#dataset
tar -xvzf /cogrob-avl-west-vol/nuScenes/compressed/v1.0-trainval_meta.tgz
apt update && apt install -y unzip
unzip /cogrob-avl-west-vol/nuScenes/compressed/can_bus.zip
unzip /cogrob-avl-west-vol/nuScenes/compressed/nuScenes-map-expansion-v1.3.zip
# clean gpu
pkill -9 -f "tools/train.py"
pkill -9 -f "torch.distributed"
pkill -9 -f "uniad_dist_train"
pkill -9 -f "python"

#wandb
python -m pip install -U pip setuptools wheel 
python -m pip install -U wandb==0.24.2
wandb login
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

#train
./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/base_track_map_front.py 1
./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/base_intent_fornt.py 8
./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/base_intent_fornt_no_map.py 8
./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/base_intent_fornt_no_interaction.py 8

#eval
./tools/uniad_dist_eval.sh ./projects/configs/stage1_track_map/base_track_map_front.py /zihan-west-vol/UniAD/projects/work_dirs/weights/stage1/0215/epoch_6.pth 2
./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/base_intent_fornt_no_map.py 8
./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/base_intent_fornt.py ./projects/work_dirs/stage2_e2e/base_intent_fornt/last_train/epoch_20.pth 1
