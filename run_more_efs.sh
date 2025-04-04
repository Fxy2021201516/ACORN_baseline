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

# 创建一个汇总文件，记录所有 efs 的 QPS 和 Recall
summary_file="${parent_dir}/summary_all_efs.txt"
echo "efs,QPS_ACORN,Recall_ACORN" > ${summary_file}

# 循环测试 efs 值
for efs in $(seq 4 1 32); do
    dir=${parent_dir}/MB${M_beta}_efs${efs}  # 在目录名中加入 efs 值
    mkdir -p ${dir}                          

    TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt

    # 运行测试，传递 efs 参数
    ./build/demos/test_acorn $N $gamma $dataset $M $M_beta $efs &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt

    TZ='America/Los_Angeles' date +"End time: %H:%M" &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt

    # 从日志文件中提取 QPS 和 Recall
    qps_acorn=$(grep "QPS:" ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt |  awk -F'QPS:' '{print $2}' | awk '{print $1}')
    recall_acorn=$(grep "Recall:" ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt |  awk '{print $NF}')

    # 将结果追加到汇总文件中
    echo "${efs},${qps_acorn},${recall_acorn}" >> ${summary_file}
done

