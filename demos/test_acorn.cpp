#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

#include <sys/time.h>

#include "../faiss/Index.h"
#include "../faiss/IndexACORN.h"
#include "../faiss/IndexFlat.h"
#include "../faiss/IndexHNSW.h"
#include "../faiss/MetricType.h"
#include "../faiss/impl/ACORN.h"
#include "../faiss/impl/HNSW.h"
#include "../faiss/index_io.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// added these
#include <arpa/inet.h>
#include <assert.h> /* assert */
#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>
#include <math.h>
#include <nlohmann/json.hpp>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <cmath> // for std::mean and std::stdev
#include <fstream>
#include <iosfwd>
#include <numeric> // for std::accumulate
#include <set>
#include <sstream> // for ostringstream
#include <thread>
#include "utils.cpp"

struct QueryResult
{
   int query_id;
   double acorn_time;
   double acorn_qps;
   float acorn_recall;
   size_t acorn_n3;
   double acorn_1_time;
   double acorn_1_qps;
   float acorn_1_recall;
   size_t acorn_1_n3;
   double filter_time;
};

// create indices for debugging, write indices to file, and get recall stats for
// all queries
int main(int argc, char *argv[])
{
   unsigned int nthreads = std::thread::hardware_concurrency();
   std::cout
       << "====================\nSTART: running TEST_ACORN for hnsw, sift data --"
       << nthreads << "cores\n"
       << std::endl;
   // printf("====================\nSTART: running MAKE_INDICES for hnsw
   // --...\n");
   double t0 = elapsed();

   int efc = 40;    // default is 40
   int efs = 16;    //  default is 16
   int k = 10;      // search parameter
   size_t d = 1152; // dimension of the vectors to index - will be overwritten
                    // by the dimension of the dataset
   int M;           // HSNW param M TODO change M back
   int M_beta;      // param for compression
   // float attr_sel = 0.001;
   // int gamma = (int) 1 / attr_sel;
   int gamma;
   int n_centroids;
   // int filter = 0;
   std::string dataset; // must be sift1B or sift1M or tripclick
   int test_partitions = 0;
   int step = 10; // 2

   std::string assignment_type = "rand";
   int alpha = 0;

   srand(0); // seed for random number generator
   int num_trials = 60;

   size_t N = 0; // N will be how many we truncate nb from sift1M to
   int generate_json;
   std::string DATA_DIR, BASE_DIR, ATTR_DATA_DIR, MY_DIS_DIR, MY_DIS_SORT_DIR;
   std::string query_path, csv_path;

   int opt;
   { // parse arguments

      N = strtoul(argv[1], NULL, 10);
      printf("N: %ld\n", N);

      gamma = atoi(argv[2]);
      printf("gamma: %d\n", gamma);

      dataset = argv[3];
      printf("dataset: %s\n", dataset.c_str());
      if (dataset != "sift1M" && dataset != "sift1M_test" &&
          dataset != "sift1B" && dataset != "tripclick" &&
          dataset != "paper" && dataset != "paper_rand2m" &&
          dataset != "words" && dataset != "MTG" && dataset != "arxiv") // TODO
      {
         printf("got dataset: %s\n", dataset.c_str());
         fprintf(stderr,
                 "Invalid <dataset>; must be a value in [sift1M, sift1B]\n");
         exit(1);
      }

      M = atoi(argv[4]);
      printf("M: %d\n", M);

      M_beta = atoi(argv[5]);
      printf("M_beta: %d\n", M_beta);

      efs = atoi(argv[6]);
      printf("efs: %d\n", efs);

      query_path = argv[7];
      printf("query_path: %s\n", query_path.c_str());

      csv_path = argv[8];
      printf("csv_path: %s\n", csv_path.c_str());

      generate_json = atoi(argv[9]);
      printf("generate_json: %d\n", generate_json);
   }

   DATA_DIR = "../ACORN_data/" + dataset;
   BASE_DIR = "../ACORN_data/" + dataset + "/" + dataset + "_base";
   ATTR_DATA_DIR = query_path;
   std::string last_value = query_path.substr(query_path.find_last_of('_') + 1);
   MY_DIS_DIR = DATA_DIR + "/results/" + dataset + "_query_" + last_value + "/my_dis";
   MY_DIS_SORT_DIR = DATA_DIR + "/results/" + dataset + "_query_" + last_value + "/my_dis_sort";

   // load metadata
   n_centroids = gamma;

   std::vector<std::vector<int>> metadata = load_ab_muti(
       dataset, gamma, assignment_type, N, BASE_DIR); // TODO:改属性数据集名称
   metadata.resize(N);
   assert(N == metadata.size());
   for (auto &inner_vector : metadata)
   {
      std::sort(inner_vector.begin(), inner_vector.end());
   }
   printf("[%.3f s] Loaded metadata, %ld attr's found\n",
          elapsed() - t0,
          metadata.size());

   size_t nq;
   float *xq;
   std::vector<std::vector<int>> aq;
   { // load query vectors and attributes
      printf("[%.3f s] Loading query vectors and attributes\n",
             elapsed() - t0);

      size_t d2;
      // xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
      bool is_base = 0;
      // load_data(dataset, is_base, &d2, &nq, xq);
      // std::string filename =get_file_name(dataset, is_base, BASE_DIR); // TODO:添加数据集名称
      std::string filename = query_path + "/" + dataset + "_query.fvecs";
      xq = fvecs_read(filename.c_str(), &d2, &nq);
      assert(d == d2 ||
             !"query does not have same dimension as expected 128");
      if (d != d2)
      {
         d = d2;
      }

      std::cout << "query vecs data loaded, with dim: " << d2 << ", nq=" << nq
                << std::endl;
      printf("[%.3f s] Loaded query vectors from %s\n",
             elapsed() - t0,
             filename.c_str());
      aq = load_aq_multi(
          dataset, n_centroids, alpha, N, ATTR_DATA_DIR); // TODO:添加数据集名称
      for (auto &inner_vector : aq)
      {
         std::sort(inner_vector.begin(), inner_vector.end());
      }
      std::cout << "aq.size():" << aq.size() << std::endl;
      printf("[%.3f s] Loaded %ld %s queries\n",
             elapsed() - t0,
             nq,
             dataset.c_str());
   }

   // create normal (base) and hybrid index
   printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d\n",
          elapsed() - t0,
          d,
          M,
          N,
          gamma);

   std::vector<QueryResult> all_query_results(nq);
   for (int i = 0; i < nq; i++)
   {
      all_query_results[i].query_id = i;
   }

   //  // base HNSW index
   //  faiss::IndexHNSWFlat base_index(d, M, 1); // gamma = 1
   //  base_index.hnsw.efConstruction = efc;     // default is 40  in HNSW.capp
   //  base_index.hnsw.efSearch = efs;           // default is 16 in HNSW.capp

   // ACORN-gamma
   faiss::IndexACORNFlat hybrid_index(d, M, gamma, metadata, M_beta);
   hybrid_index.acorn.efSearch = efs; // default is 16 HybridHNSW.capp
   debug("ACORN index created%s\n", "");

   // ACORN-1
   faiss::IndexACORNFlat hybrid_index_gamma1(d, M, 1, metadata, M * 2);
   hybrid_index_gamma1.acorn.efSearch = efs; // default is 16 HybridHNSW.capp

   { // populating the database
      std::cout << "====================Vectors====================\n"
                << std::endl;

      printf("[%.3f s] Loading database\n", elapsed() - t0);

      size_t nb, d2;
      bool is_base = 1;
      std::string filename = get_file_name(dataset, is_base, BASE_DIR); // TODO
      float *xb = fvecs_read(filename.c_str(), &d2, &nb);
      assert(d == d2 || !"dataset does not dim 128 as expected");
      printf("[%.3f s] Loaded base vectors from file: %s\n",
             elapsed() - t0,
             filename.c_str());

      std::cout << "data loaded, with dim: " << d2 << ", nb=" << nb
                << std::endl;

      printf("[%.3f s] Indexing database, size %ld*%ld from max %ld\n",
             elapsed() - t0,
             N,
             d2,
             nb);

      // index->add(nb, xb);

      printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

      //   base_index.add(N, xb);
      //   printf("[%.3f s] Vectors added to base index \n", elapsed() - t0);
      //   std::cout << "Base index vectors added: " << nb << std::endl;
      double t_acorn_0 = elapsed();
      hybrid_index.add(N, xb);
      double t_acorn_1 = elapsed();
      printf("[%.3f s] Vectors added to hybrid index \n", elapsed() - t0);
      std::cout << "Hybrid index vectors added" << nb << std::endl;
      std::cout << "========ACORN index add time: " << t_acorn_1 - t_acorn_0 << std::endl;
      //  printf("SKIPPED creating ACORN-gamma\n");
      double t_acorn1_0 = elapsed();
      hybrid_index_gamma1.add(N, xb);
      double t_acorn1_1 = elapsed();
      printf("[%.3f s] Vectors added to hybrid index with gamma=1 \n",
             elapsed() - t0);
      std::cout << "Hybrid index with gamma=1 vectors added" << nb
                << std::endl;
      std::cout << "========ACORN-1 index add time: " << t_acorn1_1 - t_acorn1_0 << std::endl;

      delete[] xb;
   }
   { // print out stats
     //   printf("====================================\n");
     //   printf("============ BASE INDEX =============\n");
     //   printf("====================================\n");
     //   base_index.printStats(false);
      printf("====================================\n");
      printf("============ ACORN INDEX =============\n");
      printf("====================================\n");
      hybrid_index.printStats(false);
   }

   printf("==============================================\n");
   printf("====================Search Results====================\n");
   printf("==============================================\n");
   double t1 = elapsed();

   //===================ACORN-gamma=========================
   { // searching the hybrid database
      printf("==================== ACORN INDEX ====================\n");
      printf("[%.3f s] Searching the %d nearest neighbors "
             "of %ld vectors in the index, efsearch %d\n",
             elapsed() - t0,
             k,
             nq,
             hybrid_index.acorn.efSearch);

      std::vector<faiss::idx_t> nns2(k * nq);
      std::vector<float> dis2(k * nq);
      std::cout << "aq.size():" << aq.size() << std::endl;
      std::cout << "nq:" << nq << std::endl;
      std::cout << "metadata.size():" << metadata.size() << std::endl;

      double t_filter_0 = elapsed();
      std::vector<char> filter_ids_map(nq * N);
      for (int xq = 0; xq < nq; xq++)
      {
         for (int xb = 0; xb < N; xb++)
         {
            const auto &query_attrs = aq[xq]; // 当前查询的属性列表（有序）
            const auto &data_attrs =
                metadata[xb]; // 当前数据库条目的属性列表（有序）

            bool is_subset = true;
            size_t i = 0, j = 0;
            const size_t query_size = query_attrs.size();
            const size_t data_size = data_attrs.size();

            while (i < query_size && j < data_size)
            {
               if (query_attrs[i] == data_attrs[j])
               {
                  i++; // 匹配成功，检查下一个查询属性
                  j++; // 继续检查 metadata 的下一个属性
               }
               else if (query_attrs[i] > data_attrs[j])
               {
                  j++; // metadata 当前值太小，往后找更大的
               }
               else
               {
                  is_subset = false; // query_attrs[i] <
                                     // data_attrs[j]，缺少该属性
                  break;
               }
            }

            // 如果 query_attrs 还没遍历完，说明 metadata 缺少某些属性
            if (i < query_size)
            {
               is_subset = false;
            }

            filter_ids_map[xq * N + xb] = is_subset;
         }
      }

      //   create filter_ids_map, ie a bitmap of the ids that are in thefilter
      //     std::vector<char> filter_ids_map(nq * N);
      //     for (int xq = 0; xq < nq; xq++) {
      //         for (int xb = 0; xb < N; xb++) {
      //             filter_ids_map[xq * N + xb] = (bool)(metadata[xb] ==
      //             aq[xq]);
      //         }
      //     }
      double t_filter_1 = elapsed();
      printf("[%.3f s] filter_ids_map created, size %ld*%ld\n",
             elapsed() - t0,
             nq,
             N);
      printf("[%.3f s] filter_ids_map creation time: %f seconds\n",
             elapsed() - t0,
             t_filter_1 - t_filter_0);

      for (auto &result : all_query_results)
      { // 保存filter时间到每个查询结果
         result.filter_time = (t_filter_1 - t_filter_0) / double(nq);
      }

      std::vector<double> query_times(nq);
      std::vector<double> query_qps(nq);
      std::vector<size_t> query_n3(nq); // 新增
      double t1_x = elapsed();
      hybrid_index.search(
          nq,
          xq,
          k,
          dis2.data(),
          nns2.data(),
          filter_ids_map.data(),
          &query_times, // 传入时间记录
          &query_qps,   // 传入QPS记录
          &query_n3     // 传入n3记录
      );
      double t2_x = elapsed();

      printf("[%.3f s] Query results (vector ids, then distances):\n",
             elapsed() - t0);
      double search_time = t2_x - t1_x;
      double qps = nq / search_time; // 核心计算
      printf("[%.3f s] *** ACORN: Query time: %f seconds, QPS: %.3f\n",
             elapsed() - t0,
             search_time,
             qps);
      // 打印每个查询的n3、时间和QPS
      for (int i = 0; i < nq; i++)
      {
         printf("Query %ld: n3=%zu, Time=%.6f s, QPS=%.3f\n",
                i, query_n3[i], query_times[i], query_qps[i]);
      }

      // 打印前10个查询结果
      int nq_print = std::min(10, (int)nq);
      for (int i = 0; i < nq_print; i++)
      {
         printf("query %2d nn's: [", i);
         for (size_t attr = 0; attr < aq[i].size(); attr++)
         {
            printf("%d%s",
                   aq[i][attr],
                   attr < aq[i].size() - 1 ? ", " : "");
         }
         printf("]: ");
         for (int j = 0; j < k; j++)
         {
            printf("%7ld [", nns2[j + i * k]);
            const auto &meta_vec = metadata[nns2[j + i * k]];
            for (size_t attr = 0; attr < meta_vec.size(); attr++)
            {
               printf("%d%s",
                      meta_vec[attr],
                      attr < meta_vec.size() - 1 ? ", " : "");
            }
            printf("] ");
         }
         printf("\n     dis: \t");
         for (int j = 0; j < k; j++)
         {
            printf("%7g ", dis2[j + i * k]);
         }
         printf("\n");
      }

      //==============计算recall==========================
      double t_recall_0 = elapsed();

      float *all_distances = new float[nq * N]; // 存储距离结果
      std::vector<std::vector<std::pair<int, float>>> sorted_results;
      if (generate_json)
      {
         hybrid_index.calculate_distances(
             nq, xq, k, all_distances, nns2.data());
         save_distances_to_txt(nq, N, all_distances, "distances", MY_DIS_DIR);
         sorted_results = get_sorted_filtered_distances(
             all_distances, filter_ids_map, nq, N);
         save_sorted_filtered_distances_to_txt(
             sorted_results,
             std::string(MY_DIS_SORT_DIR), // 输出目录
             "filter_sorted_dist_"         // 文件名前缀
         );
      }
      else
      {
         // all_distances = read_all_distances_from_txt(std::string(MY_DIS_DIR), nq, N);
         sorted_results = read_all_sorted_filtered_distances_from_txt(
             std::string(MY_DIS_SORT_DIR), nq, N);
         std::cout << "sorted_results.size():" << sorted_results.size() << std::endl;
      }

      auto recalls = compute_recall(nns2, sorted_results, nq, k);
      // 打印recall
      // for (int i = 0; i < nq; i++)
      // {
      //    printf("query %d: %.2f\n", i, recalls[i]);
      // }
      // recall平均值
      float recall_sum =
          std::accumulate(recalls.begin(), recalls.end(), 0.0f);
      float recall_mean = recall_sum / nq;
      printf("ACORN: Recall: %.2f\n", recall_mean);
      double t_recall_1 = elapsed();
      printf("[%.3f s] cal Recall time: %f seconds\n",
             elapsed() - t0,
             t_recall_1 - t_recall_0);

      for (int i = 0; i < nq; i++)
      {
         all_query_results[i].acorn_time = query_times[i];
         all_query_results[i].acorn_qps = query_qps[i];
         all_query_results[i].acorn_recall = recalls[i];
         all_query_results[i].acorn_n3 = query_n3[i];
      }

      std::cout << "finished hybrid index examples" << std::endl;
   }
   { // look at stats
      const faiss::ACORNStats &stats = faiss::acorn_stats;

      std::cout << "============= ACORN QUERY PROFILING STATS ============="
                << std::endl;
      printf("[%.3f s] Timing results for search of k=%d nearest neighbors of nq=%ld vectors in the index\n",
             elapsed() - t0,
             k,
             nq);
      std::cout << "n1: " << stats.n1 << std::endl;
      std::cout << "n2: " << stats.n2 << std::endl;
      std::cout << "n3 (number distance comps at level 0): " << stats.n3
                << std::endl;
      std::cout << "ndis: " << stats.ndis << std::endl;
      std::cout << "nreorder: " << stats.nreorder << std::endl;
      printf("average distance computations per query: %f\n",
             (float)stats.n3 / stats.n1);
   }

   // ===================ACORN-1=====================================
   { // searching the hybrid database
      printf("==================== ACORN-1 INDEX ====================\n");
      printf("[%.3f s] Searching the %d nearest neighbors "
             "of %ld vectors in the index, efsearch %d\n",
             elapsed() - t0,
             k,
             nq,
             hybrid_index.acorn.efSearch);

      std::vector<faiss::idx_t> nns3(k * nq);
      std::vector<float> dis3(k * nq);
      std::cout << "aq.size():" << aq.size() << std::endl;
      std::cout << "nq:" << nq << std::endl;

      std::vector<char> filter_ids_map3(nq * N);
      for (int xq = 0; xq < nq; xq++)
      {
         for (int xb = 0; xb < N; xb++)
         {
            const auto &query_attrs = aq[xq]; // 当前查询的属性列表（有序）
            const auto &data_attrs = metadata[xb];

            bool is_subset = true;
            size_t i = 0, j = 0;
            const size_t query_size = query_attrs.size();
            const size_t data_size = data_attrs.size();

            while (i < query_size && j < data_size)
            {
               if (query_attrs[i] == data_attrs[j])
               {
                  i++; // 匹配成功，检查下一个查询属性
                  j++; // 继续检查 metadata 的下一个属性
               }
               else if (query_attrs[i] > data_attrs[j])
               {
                  j++; // metadata 当前值太小，往后找更大的
               }
               else
               {
                  is_subset = false; // query_attrs[i] <
                                     // data_attrs[j]，缺少该属性
                  break;
               }
            }

            // 如果 query_attrs 还没遍历完，说明 metadata 缺少某些属性
            if (i < query_size)
            {
               is_subset = false;
            }

            filter_ids_map3[xq * N + xb] = is_subset;
         }
      }
      std::cout << "filter_ids_map.size():" << filter_ids_map3.size()
                << std::endl;

      std::vector<double> query_times3(nq);
      std::vector<double> query_qps3(nq);
      std::vector<size_t> query_n33(nq);
      double t1_x = elapsed();
      hybrid_index_gamma1.search(
          nq,
          xq,
          k,
          dis3.data(),
          nns3.data(),
          filter_ids_map3.data(),
          &query_times3,
          &query_qps3,
          &query_n33);
      double t2_x = elapsed();

      printf("[%.3f s] Query results (vector ids, then distances):\n",
             elapsed() - t0);
      double search_time = t2_x - t1_x;
      double qps = nq / search_time; // 核心计算
      printf("[%.3f s] *** ACORN-1: Query time: %f seconds, QPS: %.3f\n",
             elapsed() - t0,
             search_time,
             qps);
      // 打印每个查询的n3、时间和QPS
      for (int i = 0; i < nq; i++)
      {
         printf("Query %ld: n3=%zu, Time=%.6f s, QPS=%.3f\n",
                i, query_n33[i], query_times3[i], query_qps3[i]);
      }

      int nq_print = std::min(10, (int)nq);
      for (int i = 0; i < nq_print; i++)
      {
         printf("query %2d nn's: [", i);
         for (size_t attr = 0; attr < aq[i].size(); attr++)
         {
            printf("%d%s",
                   aq[i][attr],
                   attr < aq[i].size() - 1 ? ", " : "");
         }
         printf("]: ");
         for (int j = 0; j < k; j++)
         {
            printf("%7ld [", nns3[j + i * k]);
            const auto &meta_vec = metadata[nns3[j + i * k]];
            for (size_t attr = 0; attr < meta_vec.size(); attr++)
            {
               printf("%d%s",
                      meta_vec[attr],
                      attr < meta_vec.size() - 1 ? ", " : "");
            }
            printf("] ");
         }
         printf("\n     dis: \t");
         for (int j = 0; j < k; j++)
         {
            printf("%7g ", dis3[j + i * k]);
         }
         printf("\n");
      }

      //==============计算recall==========================
      auto sorted_results = read_all_sorted_filtered_distances_from_txt(
          std::string(MY_DIS_SORT_DIR), nq, N);
      std::cout << "sorted_results.size():" << sorted_results.size() << std::endl;

      auto recalls = compute_recall(nns3, sorted_results, nq, k);
      // recall平均值
      float recall_sum =
          std::accumulate(recalls.begin(), recalls.end(), 0.0f);
      float recall_mean = recall_sum / nq;
      printf("ACORN-1: Recall: %.2f\n", recall_mean);

      for (int i = 0; i < nq; i++)
      {
         all_query_results[i].acorn_1_time = query_times3[i];
         all_query_results[i].acorn_1_qps = query_qps3[i];
         all_query_results[i].acorn_1_recall = recalls[i];
         all_query_results[i].acorn_1_n3 = query_n33[i];
      }

      std::cout << "finished hybrid index examples" << std::endl;
   }
   { // look at stats
      const faiss::ACORNStats &stats = faiss::acorn_stats;

      std::cout << "============= ACORN-1 QUERY PROFILING STATS ============="
                << std::endl;
      printf("[%.3f s] Timing results for search of k=%d nearest neighbors of nq=%ld vectors in the index\n",
             elapsed() - t0,
             k,
             nq);
      std::cout << "n1: " << stats.n1 << std::endl;
      std::cout << "n2: " << stats.n2 << std::endl;
      std::cout << "n3 (number distance comps at level 0): " << stats.n3
                << std::endl;
      std::cout << "ndis: " << stats.ndis << std::endl;
      std::cout << "nreorder: " << stats.nreorder << std::endl;
      printf("average distance computations per query: %f\n",
             (float)stats.n3 / stats.n1);
   }

   std::ofstream csv_file(csv_path);
   csv_file << "QueryID,acorn_Time,acorn_QPS,acorn_Recall,acorn_n3,"
            << "ACORN_1_Time,ACORN_1_QPS,ACORN_1_Recall,ACORN_1_n3,FilterMapTime\n";

   for (const auto &result : all_query_results)
   {
      csv_file << result.query_id << ","
               << result.acorn_time << "," << result.acorn_qps << ","
               << result.acorn_recall << "," << result.acorn_n3 << ","
               << result.acorn_1_time << "," << result.acorn_1_qps << ","
               << result.acorn_1_recall << "," << result.acorn_1_n3 << ","
               << result.filter_time << "\n";
   }

   csv_file.close();
   printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
}