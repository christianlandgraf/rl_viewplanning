#include <iostream>
#include <fstream>
#include <cstdint>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/conversions.h>


#include "point_cloud_handler.h"

namespace point_cloud_handler
{

//----------------------------------------------------------------------
// CONSTRUCTORS
//----------------------------------------------------------------------

/** Construct a new PointCloudHandler object */
PointCloudHandler::PointCloudHandler(ros::NodeHandle* nodehandle) : 
    nh(*nodehandle)
{
    initializeServices();
}

//----------------------------------------------------------------------
// INITIALIZING
//----------------------------------------------------------------------


/** Initialize Services */
void PointCloudHandler::initializeServices()
{
    ROS_INFO("Initialize Services");
    voxel_grid_filter_service = nh.advertiseService("point_cloud_handler/get_area_gain",
        &PointCloudHandler::compute_voxel_grid_filter, this);
}


/**
 * @brief Surface area gain calculation based on Voxel grids. Further clean up required!
 * 
 * @param req ipa_kifz_viewplanning/srv/GetAreaGain request
 * @param res ipa_kifz_viewplanning/srv/GetAreaGain response
 * @return True if successful.
 */
bool PointCloudHandler::compute_voxel_grid_filter(
    ipa_kifz_viewplanning::GetAreaGain::Request &req, 
    ipa_kifz_viewplanning::GetAreaGain::Response &res)
{


  if(!req.standalone_pcd)
  {
    // load previously cumulated pcd and new pcd
    pcl::PointCloud<pcl::PointXYZ>::Ptr old_cum_pcd (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromROSMsg(req.previous_pcd, *old_cum_pcd);
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_cum_pcd (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromROSMsg(req.new_pcd, *new_cum_pcd);

    // get min and max values of in_pcd and clip cum_pcd for further processing
    pcl::PointXYZ min_p, max_p;
    pcl::getMinMax3D (*new_cum_pcd, min_p, max_p);
    // now use min and max for CropBox filter
    // vector for indices of points inside box
    std::vector<int> indices;
    pcl::CropBox<pcl::PointXYZ> box_filter;
    box_filter.setMin(Eigen::Vector4f(min_p.x, min_p.y, 1.07, 1.0));
    box_filter.setMax(Eigen::Vector4f(max_p.x, max_p.y, 1.5, 1.0));
    // crop new pcd
    box_filter.setInputCloud(new_cum_pcd);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_new_cum_pcd (new pcl::PointCloud<pcl::PointXYZ> ());
    box_filter.filter(*cropped_new_cum_pcd);
    // crop old cum pcd inside new pcd box
    box_filter.setInputCloud(old_cum_pcd);
    pcl::PointCloud<pcl::PointXYZ>::Ptr crop_old_cum_pcd_in_box (new pcl::PointCloud<pcl::PointXYZ> ());
    box_filter.filter(*crop_old_cum_pcd_in_box);

    // now extract all points of the old cum pcd that are located outside the bbox of the new pcd
    // these points are needed for later concatenation with processed points of new cum pcd, that are all inside the bbox)
    // first get indices of points inside box
    box_filter.filter(indices);
    
    pcl::PointIndices::Ptr inliers{new pcl::PointIndices};
    for(int point:indices)
    	inliers->indices.push_back(point);

    // and extract points outside of the bbox
    pcl::PointCloud<pcl::PointXYZ>::Ptr crop_old_cum_pcd_outside_box (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(old_cum_pcd);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*crop_old_cum_pcd_outside_box); 

    // concatenate new pcd and old pcd inside bbox (with required prior conversion to pcd2)
    pcl::PCLPointCloud2::Ptr cropped_old_cum_pcd2 (new pcl::PCLPointCloud2 ());
    pcl::toPCLPointCloud2 (*crop_old_cum_pcd_in_box, *cropped_old_cum_pcd2);
    pcl::PCLPointCloud2::Ptr cropped_new_cum_pcd2 (new pcl::PCLPointCloud2 ());
    pcl::toPCLPointCloud2 (*cropped_new_cum_pcd, *cropped_new_cum_pcd2);
    *cropped_new_cum_pcd2 += *cropped_old_cum_pcd2;
    // ...and convert back from pcd2 to pcd
    pcl::PointCloud<pcl::PointXYZ>::Ptr crop_new_cum_pcd (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromPCLPointCloud2 (*cropped_new_cum_pcd2, *crop_new_cum_pcd);
    
    // From now on process both point clouds in order to calculate area gain by the difference of both
    //Create a KD-Tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_new (new pcl::search::KdTree<pcl::PointXYZ>);

    // Output has the PointXYZ type (not PointNormal as in tutorial) in order to feed later processes
    pcl::PointCloud<pcl::PointXYZ> sm_new_pcd;

    // Init object (second point type is for the normals, even if unused)
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls_new;
    mls_new.setComputeNormals (true);
    mls_new.setInputCloud (crop_new_cum_pcd);
    mls_new.setPolynomialOrder (2);
    mls_new.setSearchMethod (tree_new);
    mls_new.setSearchRadius (0.0035);
    mls_new.process (sm_new_pcd);

    // convert XYZ to PCLPointClud2, which is the designated input for the voxel grid filter
    // pcl::PCLPointCloud2::Ptr sm_old_pcd2 (new pcl::PCLPointCloud2 ());
    // pcl::toPCLPointCloud2 (sm_old_pcd, *sm_old_pcd2);
    pcl::PCLPointCloud2::Ptr sm_new_pcd2 (new pcl::PCLPointCloud2 ());
    pcl::toPCLPointCloud2 (sm_new_pcd, *sm_new_pcd2);

    // create pointcloud for filtered output
    // pcl::PCLPointCloud2::Ptr sm_old_pcd2_filt (new pcl::PCLPointCloud2 ());
    pcl::PCLPointCloud2::Ptr sm_new_pcd2_filt (new pcl::PCLPointCloud2 ());

    // Create the filtering object, grid size in meters
    float gridsize = 0.002;
    // pcl::VoxelGrid<pcl::PCLPointCloud2> vg_old;
    // vg_old.setInputCloud (sm_old_pcd2);
    // vg_old.setLeafSize (gridsize, gridsize, gridsize);
    // vg_old.filter (*sm_old_pcd2_filt);

    pcl::VoxelGrid<pcl::PCLPointCloud2> vg_new;
    vg_new.setInputCloud (sm_new_pcd2);
    vg_new.setLeafSize (gridsize, gridsize, gridsize);
    vg_new.filter (*sm_new_pcd2_filt);

    
    // Save result in response
    pcl::PCLPointCloud2::Ptr cum_pcd2_outside_box_for_export (new pcl::PCLPointCloud2 ());
    pcl::toPCLPointCloud2 (*crop_old_cum_pcd_outside_box, *cum_pcd2_outside_box_for_export);
    *cum_pcd2_outside_box_for_export += *sm_new_pcd2_filt;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cum_pcd (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromPCLPointCloud2 (*cum_pcd2_outside_box_for_export, *cum_pcd);

    // Convert to sensor_msgs::PointCloud2 format
    pcl::toROSMsg(*cum_pcd, res.cumulated_pcd);

    float area_gain;
    area_gain = sm_new_pcd2_filt->width * sm_new_pcd2_filt->height * gridsize * gridsize - cropped_old_cum_pcd2->width * cropped_old_cum_pcd2->height * gridsize * gridsize;
    if (area_gain < 0){
      area_gain = 0;
    }
    res.area_gain = area_gain;

    // print scanned pointcloud surface area in m²
    std::cout << area_gain << std::endl;

  }
  else
  {
    // Check and convert point clouds in request
    pcl::PointCloud<pcl::PointXYZ>::Ptr old_cum_pcd (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromROSMsg(req.previous_pcd, *old_cum_pcd);


    // get min and max values of in_pcd and clip cum_pcd for further processing
    pcl::PointXYZ min_p, max_p;
    pcl::getMinMax3D (*old_cum_pcd, min_p, max_p);
    // now use min and max for CropBox filter
    pcl::CropBox<pcl::PointXYZ> box_filter;
    box_filter.setMin(Eigen::Vector4f(min_p.x, min_p.y, 1.07, 1.0));
    box_filter.setMax(Eigen::Vector4f(max_p.x, max_p.y, 1.5, 1.0));
    box_filter.setInputCloud(old_cum_pcd);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_old_cum_pcd (new pcl::PointCloud<pcl::PointXYZ> ());
    box_filter.filter(*cropped_old_cum_pcd);
    
    // Create a KD-Tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

    // Output has the PointXYZ type (not PointNormal as in tutorial) in order to feed later processes
    pcl::PointCloud<pcl::PointXYZ> sm_old_pcd; // name is confusing, as pointcloud contains no normals

    // Init object (second point type is for the normals, even if unused)
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls_old;
    mls_old.setComputeNormals (true);
    // Set parameters
    mls_old.setInputCloud (cropped_old_cum_pcd);
    mls_old.setPolynomialOrder (2);
    mls_old.setSearchMethod (tree);
    mls_old.setSearchRadius (0.0035); // (0.005)
    // Reconstruct
    mls_old.process (sm_old_pcd);

    // convert XYZ to PCLPointClud2, which is the designated input for the voxel grid filter
    pcl::PCLPointCloud2::Ptr sm_old_pcd2 (new pcl::PCLPointCloud2 ());
    pcl::toPCLPointCloud2 (sm_old_pcd, *sm_old_pcd2);

    // create pointcloud for filtered output
    pcl::PCLPointCloud2::Ptr sm_old_pcd2_filt (new pcl::PCLPointCloud2 ());

    // Create the filtering object, grid size in meters
    float gridsize = 0.002; // 0.0025;
    pcl::VoxelGrid<pcl::PCLPointCloud2> vg_old;
    vg_old.setInputCloud (sm_old_pcd2);
    vg_old.setLeafSize (gridsize, gridsize, gridsize);
    vg_old.filter (*sm_old_pcd2_filt);
    

    pcl::PointCloud<pcl::PointXYZ>::Ptr cum_pcd (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromPCLPointCloud2 (*sm_old_pcd2_filt, *cum_pcd);

    // Convert to sensor_msgs::PointCloud2 format
    pcl::toROSMsg(*cum_pcd, res.cumulated_pcd);


    // print scanned pointcloud surface area in m²
    float area_gain;
    area_gain = sm_old_pcd2_filt->width * sm_old_pcd2_filt->height * gridsize * gridsize;
    if (area_gain < 0){
      area_gain = 0;
    }
    res.area_gain = area_gain;
    std::cout << area_gain << std::endl;
  
  }


  return true;
}


} // namespace point_cloud_handler

//----------------------------------------------------------------------
// MAIN FUNCTION
//----------------------------------------------------------------------

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "point_cloud_handler"); //node name

    ros::NodeHandle nh;

    ROS_INFO("Start PointCloudHandler");
    point_cloud_handler::PointCloudHandler PointCloudHandler(&nh);  

    ros::spin();

    return 0;
} 