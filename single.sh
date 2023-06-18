#!/bin/bash
eval "$(conda shell.bash hook)"
VIDEO_PATH="/home/mert/stgcn/demo/v_SoccerJuggling_g02_c01.avi"
conda deactivate
conda activate uni
python3 /home/mert/uniformerv2_demo/app.py $VIDEO_PATH
conda deactivate
conda activate stgcn
cd stgcn
python demo/demo_skeleton.py $VIDEO_PATH demo/demo.mp4 --config configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py --checkpoint /home/mert/stgcn/configs/j.pth
cd ..
python3 /home/mert/stgcn/comma_placer.py
python3 /home/mert/uniformerv2_demo/boxtest.py $VIDEO_PATH
conda deactivate
conda activate uni
python3 /home/mert/uniformerv2_demo/appwithboundingboxes.py $VIDEO_PATH
python3 /home/mert/demo.py
