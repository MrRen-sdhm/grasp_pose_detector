#include <gpd/candidate/candidates_generator.h>

namespace gpd {
namespace candidate {

CandidatesGenerator::CandidatesGenerator(
    const Parameters &params, const HandSearch::Parameters &hand_search_params)
    : params_(params) {
  Eigen::initParallel();

  hand_search_ = std::make_unique<candidate::HandSearch>(hand_search_params);
}

void CandidatesGenerator::preprocessPointCloud(util::Cloud &cloud) {
    double t0_process = omp_get_wtime();
    printf("Processing cloud with %zu points.\n", cloud.getCloudOriginal()->size());

    // util::Plot plotter(0, 0);
    // plotter.plotCloud(cloud.getCloudOriginal(), "origion");

    // Calculate surface normals using integral images if possible.
    if (cloud.getCloudOriginal()->isOrganized() && cloud.getNormals().cols() == 0)
    {
        std::cout << "[INFO Organize] Input cloud is organized." << "\n";
        cloud.calculateNormals(0);
    }

    // Workspace filtering
    cloud.filterWorkspace(params_.workspace_);

    // plotter.plotCloud(cloud.getCloudProcessed(), "filterWorkspace");

    // Perform statistical outlier removal
    if (0)
    {
        util::Plot plotter(0, 0);
        plotter.plotCloud(cloud.getCloudProcessed(), "before");

        // Create the filtering object
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
        sor.setInputCloud(cloud.getCloudProcessed());
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*cloud.getCloudProcessed());
        std::cout << "Cloud after removing statistical outliers: " << cloud.getCloudProcessed()->size() << std::endl;
        if(cloud.getCloudOriginal()->isOrganized()) std::cout << "[INFO Organize] Cloud is organized after statistical filtering." << "\n";
        else std::cout << "[INFO Organize] Cloud is not organized after statistical filtering." << "\n";
        plotter.plotCloud(cloud.getCloudProcessed(), "after");
    }

    // Voxelization
    if (params_.voxelize_) {
        cloud.voxelizeCloud(params_.voxel_size_);

//    util::Plot plotter(0, 0);
//    plotter.plotCloud(cloud.getCloudProcessed(), "voxelizeCloud");
    }

    // Normals calculating
    if(cloud.getNormals().cols() == 0)
    {
        cloud.calculateNormals(params_.num_threads_);
    }

    // Subsample the samples above plane
    if (params_.sample_above_plane_) {
        cloud.sampleAbovePlane();
    }

    // Subsample the samples
    cloud.filterSamples(params_.workspace_);
    cloud.subsample(params_.num_samples_);

    double t_process = omp_get_wtime() - t0_process;
    printf("======== CLOUD PROCESS RUNTIME ========\n");
    printf(" TOTAL: %3.4fs\n\n", t_process);
}

void CandidatesGenerator::preprocessPointCloud(util::Cloud &cloud, cv::Rect rect) {
    double t0_process = omp_get_wtime();
    printf("Processing cloud with %zu points.\n", cloud.getCloudOriginal()->size());

    util::Plot plotter(0, 0);
//    plotter.plotCloud(cloud.getCloudOriginal(), "origion");

    // Calculate surface normals using integral images if possible.
    if (cloud.getCloudOriginal()->isOrganized() && cloud.getNormals().cols() == 0) {
        std::cout << "[INFO Organize] Input cloud is organized." << "\n";
        cloud.calculateNormals(0);
    }

    // Get samples from origin cloud
    cloud.getSamplesRegion(rect);
    plotter.plotCloud(cloud.getCloudObjRegion(), "ObjRegionCloud");
    plotter.plotCloud(cloud.getCloudObjCenter(), "ObjCenterCloud");
//    cloud.saveCloud("../ObjCenterCloud.pcd", cloud.getCloudObjCenter());

    // Workspace filtering
    cloud.filterWorkspace(params_.workspace_);
//        plotter.plotCloud(cloud.getCloudProcessed(), "filterWorkspace");

    // Voxelization
    if (params_.voxelize_) {
        cloud.voxelizeCloud(params_.voxel_size_);
//            plotter.plotCloud(cloud.getCloudProcessed(), "voxelizeCloud");
    }

    // Normals calculating
    if (cloud.getNormals().cols() == 0) {
        cloud.calculateNormals(params_.num_threads_);
    }

    // Subsample the samples above plane
    if (params_.sample_above_plane_) {
        cloud.sampleAbovePlane();
    }

    // Subsample the samples
    cloud.subsample(params_.num_samples_);

    double t_process = omp_get_wtime() - t0_process;
    printf("======== CLOUD PROCESS RUNTIME ========\n");
    printf(" TOTAL: %3.4fs\n", t_process);
}

std::vector<std::unique_ptr<Hand>> CandidatesGenerator::generateGraspCandidates(
    const util::Cloud &cloud_cam) {
  // Find sets of grasp candidates.
  std::vector<std::unique_ptr<HandSet>> hand_set_list =
      hand_search_->searchHands(cloud_cam);
  printf("Evaluated %d hand sets with %d potential hand poses.\n",
         (int)hand_set_list.size(),
         (int)(hand_set_list.size() * hand_set_list[0]->getHands().size()));

  // Extract the grasp candidates.
  std::vector<std::unique_ptr<Hand>> candidates;
  for (int i = 0; i < hand_set_list.size(); i++) {
    for (int j = 0; j < hand_set_list[i]->getHands().size(); j++) {
      if (hand_set_list[i]->getIsValid()(j)) {
        candidates.push_back(std::move(hand_set_list[i]->getHands()[j]));
      }
    }
  }
  std::cout << "Generated " << candidates.size() << " grasp candidates.\n";

  return candidates;
}

std::vector<std::unique_ptr<HandSet>>
CandidatesGenerator::generateGraspCandidateSets(const util::Cloud &cloud_cam) {
  // Find sets of grasp candidates.
  std::vector<std::unique_ptr<HandSet>> hand_set_list =
      hand_search_->searchHands(cloud_cam);

  return hand_set_list;
}

std::vector<int> CandidatesGenerator::reevaluateHypotheses(
    const util::Cloud &cloud, std::vector<std::unique_ptr<Hand>> &grasps) {
  return hand_search_->reevaluateHypotheses(cloud, grasps);
}

}  // namespace candidate
}  // namespace gpd
