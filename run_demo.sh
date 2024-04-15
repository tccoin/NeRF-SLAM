# cd /home/junzhe/slam_lib/NeRF-SLAM/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#profile
# nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --cudabacktrace=memory --capture-range=cudaProfilerApi -o=test --stats=true python3 ./examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/Replica/office0 --dataset_name=replica --buffer=50 --slam --parallel_run --img_stride=2 --fusion=''

# python3 ./examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/Replica/office0 --dataset_name=replica --buffer=50 --slam --parallel_run --img_stride=2 --fusion=''

# slam+nerf

# python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/kitti_odom/05 --dataset_name=kitti_odom --buffer=1500 --img_stride=2 --fusion='nerf' --gui --width=1920 --height=1080 --mask_type=no_depth --slam #--eval --final_k=2000 --mask_type=ours_w_thresh --slam

# python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/underwater_cave --dataset_name=underwater_cave --buffer=100 --img_stride=1 --fusion='nerf' --gui --width=1920 --height=1080 --final_k=500 --stop_iters=30000 --slam # --eval  --mask_type=ours_w_thresh --slam --mask_type=no_depth


# python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/Replica/office0 --dataset_name=replica --buffer=50 --img_stride=1 --fusion='nerf' --gui --width=1920 --height=1080 #--eval --final_k=2000 --mask_type=ours_w_thresh --slam

# python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/tartanair/seq/soulcity/Easy/P001 --dataset_name=tartanair --buffer=500 --img_stride=2  --fusion='nerf' --gui --width=1280 --height=960 --stop_iters=35000 --slam #--eval --final_k=2000 --mask_type=ours_w_thresh  --parallel_run

while true; do
python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/nerf-cube-diorama-dataset/room --dataset_name=nerf --buffer=100 --img_stride=1  --fusion='nerf' --gui --width=1280 --height=960 --stop_iters=10000 --slam --eval --mask_type=ours_w_thresh # --eval --final_k=2000 --mask_type=ours_w_thresh  --parallel_run --mask_type=no_depth
done

# droid slam
# python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/kitti_odom/05 --dataset_name=kitti_odom --buffer=1500 --img_stride=2 --fusion='' --width=1920 --height=1080 --slam #--eval --final_k=2000 --mask_type=ours_w_thresh
