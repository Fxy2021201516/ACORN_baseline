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

// create indices for debugging, write indices to file, and get recall stats for
// all queries
int main(int argc, char* argv[]) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout
            << "====================\nSTART: running TEST_ACORN for hnsw, sift data --"
            << nthreads << "cores\n"
            << std::endl;
    // printf("====================\nSTART: running MAKE_INDICES for hnsw
    // --...\n");
    double t0 = elapsed();

    int efc = 40;   // default is 40
    int efs = 16;   //  default is 16
    int k = 10;     // search parameter
    size_t d = 128; // dimension of the vectors to index - will be overwritten
                    // by the dimension of the dataset
    int M;          // HSNW param M TODO change M back
    int M_beta;     // param for compression
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
    bool generate_json = false;

    int opt;
    { // parse arguments

        if (argc < 6 || argc > 8) {
            fprintf(stderr,
                    "Syntax: %s <number vecs> <gamma> [<assignment_type>] [<alpha>] <dataset> <M> <M_beta>\n",
                    argv[0]);
            exit(1);
        }

        N = strtoul(argv[1], NULL, 10);
        printf("N: %ld\n", N);

        gamma = atoi(argv[2]);
        printf("gamma: %d\n", gamma);

        dataset = argv[3];
        printf("dataset: %s\n", dataset.c_str());
        if (dataset != "sift1M" && dataset != "sift1M_test" &&
            dataset != "sift1B" && dataset != "tripclick" &&
            dataset != "paper" && dataset != "paper_rand2m" &&
            dataset != "words") {
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
    }

    // load metadata
    n_centroids = gamma;

    std::vector<std::vector<int>> metadata = load_ab_muti(
            dataset, gamma, assignment_type, N); // TODO:改属性数据集名称
    metadata.resize(N);
    assert(N == metadata.size());
    for (auto& inner_vector : metadata) {
        std::sort(inner_vector.begin(), inner_vector.end());
    }
    printf("[%.3f s] Loaded metadata, %ld attr's found\n",
           elapsed() - t0,
           metadata.size());

    size_t nq;
    float* xq;
    std::vector<std::vector<int>> aq;
    { // load query vectors and attributes
        printf("[%.3f s] Loading query vectors and attributes\n",
               elapsed() - t0);

        size_t d2;
        // xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
        bool is_base = 0;
        // load_data(dataset, is_base, &d2, &nq, xq);
        std::string filename =
                get_file_name(dataset, is_base); // TODO:添加数据集名称
        xq = fvecs_read(filename.c_str(), &d2, &nq);
        assert(d == d2 ||
               !"query does not have same dimension as expected 128");
        if (d != d2) {
            d = d2;
        }

        std::cout << "query vecs data loaded, with dim: " << d2 << ", nq=" << nq
                  << std::endl;
        printf("[%.3f s] Loaded query vectors from %s\n",
               elapsed() - t0,
               filename.c_str());
        aq = load_aq_multi(
                dataset, n_centroids, alpha, N); // TODO:添加数据集名称
        for (auto& inner_vector : aq) {
            std::sort(inner_vector.begin(), inner_vector.end());
        }
        std::cout << "aq.size():" << aq.size() << std::endl;
        printf("[%.3f s] Loaded %ld %s queries\n",
               elapsed() - t0,
               nq,
               dataset.c_str());
    }
    //  // nq = 1;
    //  int gt_size = 100;
    //  if (dataset == "sift1M_test" || dataset == "paper") {
    //      gt_size = 10;
    //  }
    //  std::vector<faiss::idx_t> gt(gt_size * nq);
    //  { // load ground truth
    //      gt = load_gt(dataset, gamma, alpha, assignment_type, N);
    //      printf("[%.3f s] Loaded ground truth, gt_size: %d\n",
    //             elapsed() - t0,
    //             gt_size);
    //  }

    // create normal (base) and hybrid index
    printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d\n",
           elapsed() - t0,
           d,
           M,
           N,
           gamma);
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
        // printf("====================Vectors====================\n");

        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        bool is_base = 1;
        std::string filename = get_file_name(dataset, is_base);
        float* xb = fvecs_read(filename.c_str(), &d2, &nb);
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

        hybrid_index.add(N, xb);
        printf("[%.3f s] Vectors added to hybrid index \n", elapsed() - t0);
        std::cout << "Hybrid index vectors added" << nb << std::endl;
        // printf("SKIPPED creating ACORN-gamma\n");

        hybrid_index_gamma1.add(N, xb);
        printf("[%.3f s] Vectors added to hybrid index with gamma=1 \n",
               elapsed() - t0);
        std::cout << "Hybrid index with gamma=1 vectors added" << nb
                  << std::endl;

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

        //   create filter_ids_map, ie a bitmap of the ids that are in the
        //   filter
        //     std::vector<char> filter_ids_map(nq * N);
        //     for (int xq = 0; xq < nq; xq++) {
        //         for (int xb = 0; xb < N; xb++) {
        //             filter_ids_map[xq * N + xb] = (bool)(metadata[xb] ==
        //             aq[xq]);
        //         }
        //     }
        std::cout << "aq.size():" << aq.size() << std::endl;
        std::cout << "nq:" << nq << std::endl;
        std::cout << "metadata.size():" << metadata.size() << std::endl;
        //   std::vector<char> filter_ids_map(nq * N);
        //   for (int xq = 0; xq < nq; xq++) {
        //       for (int xb = 0; xb < N; xb++) {
        //           // Check if all elements in aq[xq] are present in
        //           metadata[xb] bool contains_all = true; for (int attr :
        //           aq[xq]) {
        //               if (std::find(
        //                           metadata[xb].begin(),
        //                           metadata[xb].end(),
        //                           attr) == metadata[xb].end()) {
        //                   contains_all = false;
        //                   break;
        //               }
        //           }
        //           filter_ids_map[xq * N + xb] = contains_all;
        //       }
        //   }

        double t1_x = elapsed();
        std::vector<char> filter_ids_map(nq * N);
        for (int xq = 0; xq < nq; xq++) {
            for (int xb = 0; xb < N; xb++) {
                const auto& query_attrs = aq[xq]; // 当前查询的属性列表（有序）
                const auto& data_attrs =
                        metadata[xb]; // 当前数据库条目的属性列表（有序）

                bool is_subset = true;
                size_t i = 0, j = 0;
                const size_t query_size = query_attrs.size();
                const size_t data_size = data_attrs.size();

                while (i < query_size && j < data_size) {
                    if (query_attrs[i] == data_attrs[j]) {
                        i++; // 匹配成功，检查下一个查询属性
                        j++; // 继续检查 metadata 的下一个属性
                    } else if (query_attrs[i] > data_attrs[j]) {
                        j++; // metadata 当前值太小，往后找更大的
                    } else {
                        is_subset = false; // query_attrs[i] <
                                           // data_attrs[j]，缺少该属性
                        break;
                    }
                }

                // 如果 query_attrs 还没遍历完，说明 metadata 缺少某些属性
                if (i < query_size) {
                    is_subset = false;
                }

                filter_ids_map[xq * N + xb] = is_subset;
            }
        }
        std::cout << "filter_ids_map.size():" << filter_ids_map.size()
                  << std::endl;

        hybrid_index.search(
                nq,
                xq,
                k,
                dis2.data(),
                nns2.data(),
                filter_ids_map.data()); // TODO change first argument back to nq
        double t2_x = elapsed();

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        int nq_print = std::min(100, (int)nq);
        for (int i = 0; i < nq_print; i++) {
            printf("query %2d nn's: [", i);
            for (size_t attr = 0; attr < aq[i].size(); attr++) {
                printf("%d%s",
                       aq[i][attr],
                       attr < aq[i].size() - 1 ? ", " : "");
            }
            printf("]: ");
            for (int j = 0; j < k; j++) {
                printf("%7ld [", nns2[j + i * k]);
                const auto& meta_vec = metadata[nns2[j + i * k]];
                for (size_t attr = 0; attr < meta_vec.size(); attr++) {
                    printf("%d%s",
                           meta_vec[attr],
                           attr < meta_vec.size() - 1 ? ", " : "");
                }
                printf("] ");
            }
            printf("\n     dis: \t");
            for (int j = 0; j < k; j++) {
                printf("%7g ", dis2[j + i * k]);
            }
            printf("\n");
        }

        double search_time = t2_x - t1_x;
        double qps = nq / search_time; // 核心计算

        printf("[%.3f s] *** Query time: %f seconds, QPS: %.3f\n",
               elapsed() - t0,
               search_time,
               qps);

        //==============计算recall==========================
        float* all_distances = new float[nq * N]; // 存储距离结果
        if (generate_json) {
            hybrid_index.calculate_distances(
                    nq, xq, k, all_distances, nns2.data());
            save_distances_to_txt(nq, N, all_distances, "distances");
        } else {
            all_distances =
                    read_all_distances_from_txt(std::string(MY_DIS_DIR), nq, N);
        }

        auto sorted_results = get_sorted_filtered_distances(
                all_distances, filter_ids_map, nq, N);
        save_sorted_filtered_distances_to_txt(
                sorted_results,
                std::string(MY_DIS_SORT_DIR), // 输出目录
                "filter_sorted_dist_"         // 文件名前缀
        );
        auto recalls = compute_recall(nns2, sorted_results, nq, k);
        // 打印recall
        for (int i = 0; i < nq; i++) {
            printf("query %d: %.2f\n", i, recalls[i]);
        }
        // recall平均值
        float recall_sum =
                std::accumulate(recalls.begin(), recalls.end(), 0.0f);
        float recall_mean = recall_sum / nq;
        printf("Recall: %.2f\n", recall_mean);

        std::cout << "finished hybrid index examples" << std::endl;
    }
    { // look at stats
        // const faiss::HybridHNSWStats& stats = index.hnsw_stats;
        const faiss::ACORNStats& stats = faiss::acorn_stats;

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
    /*{ // searching the base database
             printf("====================HNSW INDEX====================\n");
             printf("[%.3f s] Searching the %d nearest neighbors "
                    "of %ld vectors in the index, efsearch %d\n",
                    elapsed() - t0,
                    k,
                    nq,
                    base_index.hnsw.efSearch);

             std::vector<faiss::idx_t> nns(k * nq);
             std::vector<float> dis(k * nq);

             std::cout << "here1" << std::endl;
             std::cout << "nn and dis size: " << nns.size() << " " << dis.size()
                       << std::endl;

             double t1 = elapsed();
             base_index.search(nq, xq, k, dis.data(), nns.data());
             double t2 = elapsed();

             printf("[%.3f s] Query results (vector ids, then distances):\n",
                    elapsed() - t0);

             // take max of 5 and nq
             int nq_print = std::min(5, (int)nq);
             for (int i = 0; i < nq_print; i++) {
                 printf("query %2d nn's: ", i);
                 for (int j = 0; j < k; j++) {
                     // printf("%7ld (%d) ", nns[j + i * k], metadata.size());
                     printf("%7ld (%d) ", nns[j + i * k], metadata[nns[j + i *
                     k]]);
                 }
                 printf("\n     dis: \t");
                 for (int j = 0; j < k; j++) {
                     printf("%7g ", dis[j + i * k]);
                 }
                 printf("\n");
                 // exit(0);
             }

             printf("[%.3f s] *** Query time: %f\n", elapsed() - t0, t2 - t1);

             // print number of distance computations
             // printf("[%.3f s] *** Number of distance computations: %ld\n",
             //    elapsed() - t0, base_index.ntotal * nq);
             std::cout << "finished base index examples" << std::endl;
         }

         { // look at stats
             // const faiss::HybridHNSWStats& stats = index.hnsw_stats;
             // const faiss::HNSWStats& stats = faiss::hnsw_stats;
             const faiss::HNSWStats& stats = base_index.hnsw.hnsw_stats;

             std::cout
                     << "============= BASE HNSW QUERY PROFILING STATS
                     ============="
                     << std::endl;
             printf("[%.3f s] Timing results for search of k=%d nearest
       neighbors of nq=%ld vectors in the index\n", elapsed() - t0, k, nq);
             std::cout << "n1: " << stats.n1 << std::endl;
             std::cout << "n2: " << stats.n2 << std::endl;
             std::cout << "n3 (number distance comps at level 0): " << stats.n3
                       << std::endl;
             std::cout << "ndis: " << stats.ndis << std::endl;
             std::cout << "nreorder: " << stats.nreorder << std::endl;
             printf("average distance computations per query: %f\n",
                    (float)stats.n3 / stats.n1);
         }*/

    printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
}