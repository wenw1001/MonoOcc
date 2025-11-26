set -e
exeFunc(){
    num_seq=$1
    python utils/depth2lidar.py --calib_dir  /home/rvl/Desktop/wenwen/my_projects/MonoOcc/kitti/dataset/sequences/$num_seq \
    --depth_dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/preprocess/mobilestereonet/depth/sequences/$num_seq \
    --save_dir /home/rvl/Desktop/wenwen/my_projects/MonoOcc/preprocess/mobilestereonet/lidar/sequences/$num_seq

    cp data_odometry_calib/sequences/$num_seq/calib.txt /home/rvl/Desktop/wenwen/my_projects/MonoOcc/preprocess/mobilestereonet/lidar/sequences/$num_seq/
    cp data_odometry_calib/sequences/$num_seq/poses.txt /home/rvl/Desktop/wenwen/my_projects/MonoOcc/preprocess/mobilestereonet/lidar/sequences/$num_seq/
}

# mkdir -p $data_path/lidar
# ln -s $data_path/lidar ./mobilestereonet/lidar
for i in {00..21}
do
    exeFunc $i
done
