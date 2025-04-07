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

# 循环运行10次测试
for i in {1..5}; do
    # 为每次运行创建单独的子目录
    dir=${parent_dir}/MB${M_beta}_query${i}
    mkdir -p ${dir}
    
    # 设置当前查询路径
    query_path="../ACORN_data/words/words_query/word_query_${i}"
    
    TZ='America/Los_Angeles' date +"Start time for query ${i}: %H:%M" &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt
    
    # 运行测试程序，传入查询路径作为额外参数
    ./build/demos/test_acorn $N $gamma $dataset $M $M_beta $efs "${query_path}" &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt
    
done