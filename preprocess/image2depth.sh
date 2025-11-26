set -e
exeFunc(){
    cd mobilestereonet
    baseline=$1
    num_seq=$2
    CUDA_VISIBLE_DEVICES=0 python prediction.py --datapath ../kitti/dataset/sequences/$num_seq \
    --testlist ./filenames/$num_seq.txt --num_seq $num_seq --loadckpt ./MSNet3D_SF_DS_KITTI2015.ckpt --dataset kitti \
    --model MSNet3D --savepath "./depth" --baseline $baseline
    cd ..
}

# Change data_path to your own specified path
# And make sure there is enough space under data_path to store the generated data
# data_path=/mnt/NAS/data/yiming/segformer3d_data
data_path=./preprocess

# mkdir -p $data_path/depth
# mkdir -p  /mobilestereonet/depth
for i in 00 01 02
do
    exeFunc 388.1823 $i     
done

for i in 03
do
    exeFunc 389.6304 $i     
done

for i in 04 05 06 07 08 09 10 11 12
do
    exeFunc 381.8293 $i     
done

for i in 13 14 15 16 17 18 19 20 21
do
    exeFunc 388.1823 $i     
done
