export debugSearchFlag=1
#! /bin/bash

# 删除 build 目录及其下的文件
rm -rf build

cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build
make -C build -j faiss
make -C build utils
make -C build test_acorn

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

N=8000
gamma=80
dataset=words
M=32 
M_beta=64
efs=10


parent_dir=../ACORN_data/${dataset}/${now}_${dataset}  

rm -rf ../ACORN_data/${dataset}/${now}_${dataset}  

mkdir -p ${parent_dir}                      
dir=${parent_dir}/MB${M_beta}
mkdir -p ${dir}                          


TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt


./build/demos/test_acorn $N $gamma $dataset $M $M_beta $efs  &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt