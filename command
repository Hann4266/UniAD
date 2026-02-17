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

kill -9 -f "python"
CUDA_VISIBLE_DEVICES=1 ./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/base_track_map_front.py 1
CUDA_VISIBLE_DEVICES=1 ./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/base_track_map.py 1
eval:
./tools/uniad_dist_eval.sh ./projects/configs/stage1_track_map/base_track_map_front.py /zihan-west-vol/UniAD/projects/work_dirs/weights/stage1/0215/epoch_6.pth 2
/zihan-west-vol/Uniad_FrontBEV/UniAD/projects/mmdet3d_plugin/datasets/eval_utils/nuscenes_eval.py