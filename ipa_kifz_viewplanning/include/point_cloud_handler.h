#ifndef POINT_CLOUD_HANDLER_H_
#define POINT_CLOUD_HANDLER_H_

#include <ros/ros.h>

#include <ipa_kifz_viewplanning/GetAreaGain.h>
#include <string>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>


namespace point_cloud_handler
{

class PointCloudHandler
{
public:
    PointCloudHandler(ros::NodeHandle* nodehandle);
    //~PointCloudHandler();

private:
    ros::NodeHandle nh;
    ros::ServiceServer voxel_grid_filter_service;

    void initializeServices();

    bool compute_voxel_grid_filter(
        ipa_kifz_viewplanning::GetAreaGain::Request &req, 
        ipa_kifz_viewplanning::GetAreaGain::Response &res
    );

};

} // end namespace point_cloud_handler


#endif