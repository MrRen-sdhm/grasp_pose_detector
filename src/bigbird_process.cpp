#include <gpd/data_generator.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace gpd {
    namespace apps {
        namespace generate_data {

            int DoMain(int argc, char **argv) {
                // Read arguments from command line.
                if (argc < 3) {
                    std::cout << "Error: Not enough input arguments!\n\n";
                    std::cout << "Usage: generate_data CONFIG_FILE MODE OBJECT_NAME\n\n";
                    std::cout << "MODE: 0 generate data / MODE: 1 create MultiViewCloud\n\n";
                    std::cout << "Generate data using parameters from CONFIG_FILE (*.cfg) or Create multi view cloud.\n\n";
                    return (-1);
                }

                // Seed the random number generator.
                std::srand(std::time(0));

                // Read path to config file.
                std::string config_filename = argv[1];

                // Create training data.
                DataGenerator generator(config_filename);

                // 创建hdf5格式数据集 Hdf5Datasets
                if(!strcmp(argv[2], "0")) {
                    printf("\033[0;36m%s\033[0m\n", "[Mode] Generate data bigbird.");
                    generator.generateDataBigbird();
                }
                // 创建多视角点云 MultiViewCloud
                else if(!strcmp(argv[2], "1")) {
                    printf("\033[0;36m%s\033[0m\n", "[Mode] Create multi view cloud.");

                    // 转盘角度
                    std::vector<int> angles;
                    for (int i = 0; i < 120; i++) {
                        angles.push_back(i * 3);
                    }
//                    idx.push_back(1);

//                    int camera = 1;
                    int reference_camera = 5;
                    // 要处理的物体
                    std::string obj_names[]={"red_bull", "advil_liqui_gels", "band_aid_clear_strips", "blue_clover_baby_toy",
                                             "bumblebee_albacore", "campbells_soup_at_hand_creamy_tomato", "colgate_cool_mint", "crayola_24_crayons",
                                             "crest_complete_minty_fresh", "dove_beauty_cream_bar", "expo_marker_red", "haagen_dazs_butter_pecan",
                                             "hunts_paste", "krylon_short_cuts", "v8_fusion_peach_mango", "zilla_night_black_heat"};
//                    std::string obj_names[]={"expo_marker_red"};
                    size_t obj_cnt = sizeof(obj_names) / sizeof(obj_names[0]);
                    printf("\033[0;36m%s\033[0m\033[0;36m%zu\033[0m\033[0;36m%s\033[0m\n", "[INFO] Will create multi view cloud of ",
                           obj_cnt, " objects.");
                    // 创建各物体的多视角点云
                    for(int i = 0; i < obj_cnt; i++) {
                        for(int camera = 1; camera <= 5; camera++) {
                            util::Cloud cloud = generator.createMultiViewCloud(
                                    obj_names[i], camera, angles, reference_camera);
//                         pcl::io::savePCDFileASCII("test_pcd.pcd", *cloud.getCloudOriginal());
                        }
                        printf("\033[0;36m%s\033[0m\033[0;32m%s\033[0m\n\n", "[INFO] Created multi view cloud for ", obj_names[i].c_str());
                    }
                    printf("\033[0;36m%s\033[0m\033[0;36m%zu\033[0m\033[0;36m%s\033[0m\n", "[INFO] Created multi view cloud of ",
                           obj_cnt, " objects.");
                }


                //  cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open("test.hdf5");
                //  std::string IMAGE_DS_NAME = "images";
                //  if (!h5io->hlexists(IMAGE_DS_NAME))
                //  {
                //    int n_dims = 3;
                //    int dsdims[n_dims] = { 505000, 60, 60 };
                //    int chunks[n_dims] = {  10000, 60, 60 };
                //    cv::Mat images(n_dims, dsdims, CV_8UC(15), cv::Scalar(0.0));
                //
                //    printf("Creating dataset <images> ...\n");
                //    h5io->dscreate(n_dims, dsdims, CV_8UC(15), IMAGE_DS_NAME, 9, chunks);
                //    printf("Writing dataset <images> ...\n");
                //    h5io->dswrite(images, IMAGE_DS_NAME);
                //  }
                //  h5io->close();

                return 0;
            }

        }  // namespace generate_data
    }  // namespace apps
}  // namespace gpd

int main(int argc, char *argv[]) {
    return gpd::apps::generate_data::DoMain(argc, argv);
}
