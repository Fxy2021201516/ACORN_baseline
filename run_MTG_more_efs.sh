export debugSearchFlag=1
#! /bin/bash

# Clean build_MTG
rm -rf build_MTG

# Build with CMake
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build_MTG
make -C build_MTG -j faiss
make -C build_MTG utils
make -C build_MTG test_acorn

##########################################
# TEST CONFIGURATION
##########################################
now=$(date +"%m-%d-%Y")

N=40274
gamma=80
dataset=MTG
M=32 
M_beta=64

parent_dir=../ACORN_data/${dataset}/${now}_${dataset}  
rm -rf ../ACORN_data/${dataset}/${now}_${dataset}  
mkdir -p ${parent_dir}

# Files for raw and averaged results
raw_results="${parent_dir}/raw_results_all_efs.csv"
avg_results="${parent_dir}/averaged_results_by_efs.csv"

# Write headers
echo "query,efs,run,QPS_ACORN,Recall_ACORN,QPS_ACORN_1,Recall_ACORN_1" > ${raw_results}
echo "efs,Avg_QPS_ACORN,Avg_Recall_ACORN,Avg_QPS_ACORN_1,Avg_Recall_ACORN_1" > ${avg_results}

##########################################
# RUN TESTS
##########################################
for i in {1..1}; do
    query_path="../ACORN_data/MTG/MTG_query/MTG_query_${i}"
    # Sort efs values to ensure consistent ordering
    efs_values=($(seq 16 16 16))

    for efs in "${efs_values[@]}"; do
        dir=${parent_dir}/MB${M_beta}_query${i}_efs${efs}
        mkdir -p ${dir}

        # Run each efs 10 times
        for run in {1..5}; do
            # Set generate_json flag (true only for first efs of each query)
            if [ "$efs" -eq 16 ] && [ "$run" -eq 0 ]; then
               generate_json="1"
               echo "Generating JSON for query ${i} (first run only)"
            else
               generate_json="0"
            fi
            echo "Running test for query ${i} with efs=${efs}, run=${run}, generate_json=${generate_json}"
            ./build_MTG/demos/test_acorn $N $gamma $dataset $M $M_beta $efs "${query_path}" "${generate_json}" &>> ${dir}/summary_run${run}.txt

            # Extract metrics
            qps_acorn=$(grep "ACORN:" ${dir}/summary_run${run}.txt | grep "QPS:" | grep -v "ACORN-1" | awk -F'QPS:' '{print $2}' | awk '{print $1}')
            recall_acorn=$(grep "ACORN:" ${dir}/summary_run${run}.txt | grep "Recall:" | grep -v "ACORN-1" | awk '{print $NF}')
            qps_acorn1=$(grep "ACORN-1:" ${dir}/summary_run${run}.txt | grep "QPS:" | awk -F'QPS:' '{print $2}' | awk '{print $1}')
            recall_acorn1=$(grep "ACORN-1:" ${dir}/summary_run${run}.txt | grep "Recall:" | awk '{print $NF}')

            # Save raw results
            echo "${i},${efs},${run},${qps_acorn},${recall_acorn},${qps_acorn1},${recall_acorn1}" >> ${raw_results}
        done
    done
done

# ##########################################
# # CALCULATE AVERAGES
# ##########################################
# echo "Calculating averages for each efs value..."

# # Process each unique efs value
# for efs in $(seq 16 16 128); do
#     # Extract all lines for this efs value (all queries and all runs)
#     grep ",${efs}," ${raw_results} > ${parent_dir}/temp_efs${efs}.csv
    
#     # Calculate averages using awk - now averaging over 10 queries * 10 runs = 100 data points per efs
#     awk -F',' -v efs=${efs} '
#     BEGIN {
#         sum_qps=0; sum_recall=0; sum_qps1=0; sum_recall1=0; count=0
#     }
#     {
#         sum_qps += $4; sum_recall += $5
#         sum_qps1 += $6; sum_recall1 += $7
#         count++
#     }
#     END {
#         avg_qps = sum_qps/count
#         avg_recall = sum_recall/count
#         avg_qps1 = sum_qps1/count
#         avg_recall1 = sum_recall1/count
#         printf "%d,%.2f,%.4f,%.2f,%.4f\n", efs, avg_qps, avg_recall, avg_qps1, avg_recall1
#     }' ${parent_dir}/temp_efs${efs}.csv >> ${avg_results}
    
#     rm ${parent_dir}/temp_efs${efs}.csv
# done

# echo "All tests completed. Results saved to:"
# echo "Raw data: ${raw_results}"
# echo "Averaged results: ${avg_results}"