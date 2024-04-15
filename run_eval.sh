# seq=nerf-cube-diorama-dataset/room
# echo "Processing ${seq}"
# python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/${seq} --dataset_name=nerf --output=output/${seq} --buffer=20 --img_stride=1 --fusion='nerf' --gui --width=1920 --height=1080 --final_k=2000 --stop_iters=30000 --slam --eval

# for seq in nerf-cube-diorama-dataset/room nerf-cube-diorama-dataset/room_underwater_ugan nerf-cube-diorama-dataset/room_underwater
# do
#     echo "Processing ${seq}"
#     python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/${seq} --dataset_name=nerf --output=output/${seq} --buffer=200 --img_stride=1 --fusion='nerf' --gui --width=1920 --height=1080 --final_k=2000 --stop_iters=30000 --slam --eval --mask_type=ours_w_thresh
# done

# underwater cave, no_depth
# for seq in underwater_cave_ugan/segment12imgs underwater_cave/segment12imgs
# do
#     echo "Processing ${seq}"
#     python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/${seq} --dataset_name=underwater_cave --output=output/${seq}_no_depth --buffer=250 --img_stride=1 --fusion='nerf' --gui --width=1920 --height=1080 --final_k=2000 --stop_iters=30000 --slam --eval --mask_type=no_depth
# done

# underwater cave, droid slam depth
# for seq in underwater_cave_ugan/segment12imgs underwater_cave/segment12imgs
# do
#     echo "Processing ${seq}"
#     python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/${seq} --dataset_name=underwater_cave --output=output/${seq} --buffer=250 --img_stride=1 --fusion='nerf' --gui --width=1920 --height=1080 --final_k=2000 --stop_iters=30000 --slam --eval
# done

# bus
for seq in underwater/bus_undistorted underwater/busUGAN
do
    echo "Processing ${seq}"
    python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/${seq} --dataset_name=rgb --output=output/${seq} --buffer=200 --img_stride=20 --width=1920 --height=1080 --final_k=4000 --stop_iters=30000 --slam --eval # --gui --fusion='nerf'
done

#cemetery
for seq in underwater/cemetery_undistorted
do
    echo "Processing ${seq}"
    python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/${seq} --dataset_name=rgb --output=output/${seq} --buffer=200 --img_stride=20 --width=1920 --height=1080 --final_k=4000 --stop_iters=30000 --slam --eval
done

# python3 examples/slam_demo.py --dataset_dir=/home/junzhe/dataset/Replica/office0 --dataset_name=nerf --output=output/Replica/office0 --buffer=300 --img_stride=10 --fusion='nerf' --gui --width=1920 --height=1080 --final_k=2000 --stop_iters=30000 --slam --eval