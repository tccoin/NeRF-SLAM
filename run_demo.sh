# cd /home/junzhe/slam_lib/NeRF-SLAM/

#profile
# nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --cudabacktrace=memory --capture-range=cudaProfilerApi -o=test --stats=true python3 ./examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/Replica/office0 --dataset_name=replica --buffer=50 --slam --parallel_run --img_stride=2 --fusion=''

# python3 ./examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/Replica/office0 --dataset_name=replica --buffer=50 --slam --parallel_run --img_stride=2 --fusion=''

#eval
# python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/Replica/office0 --dataset_name=replica --buffer=300 --slam --img_stride=2 --fusion='nerf' --gui --eval --width=1920 --height=1080
python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/kitti_odom/05 --dataset_name=kitti_odom --buffer=1000 --slam --img_stride=2 --fusion='nerf' --gui --width=1920 --height=1080 --mask_type=ours_w_thresh #--eval --final_k=2000
