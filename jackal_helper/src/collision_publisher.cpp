//
// Created by phil on 24/01/18.
//

#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/gazebo_config.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Bool.h>
#include <iostream>
#include <vector>

ros::Publisher pub;
bool airborne;
const std::string DELIMITER = "::";

// Forces callback function
void forcesCb(ConstContactsPtr &_msg){
    // What to do when callback
    for (int i = 0; i < _msg->contact_size(); ++i) {

        std::string entity1 = _msg->contact(i).collision1();
        entity1 = entity1.substr(0, entity1.find(DELIMITER)); // Extract entity1 name

        std::string entity2 = _msg->contact(i).collision2();
        entity2 = entity2.substr(0, entity2.find(DELIMITER)); // Extract entity1 name

        if(entity1 != "ground_plane" && entity2 != "ground_plane"){
            if (entity1 == "jackal" || entity2 == "jackal"){
                std_msgs::Bool collide;
                collide.data = true;
                pub.publish(collide);
                ROS_INFO_STREAM(entity1 + ":" + entity2);
                return;
            }
        }
    }
}

// Position callback function
void positionCb(const nav_msgs::Odometry::ConstPtr& msg2){
    if (msg2->pose.pose.position.z > 0.3) {
        airborne = true;
    } else {
        airborne = false;
    }
}

int main(int _argc, char **_argv){
    // Set variables
    airborne = false;

    // Load Gazebo & ROS
    gazebo::client::setup(_argc, _argv);
    ros::init(_argc, _argv, "force_measure");

    // Create Gazebo node and init
    gazebo::transport::NodePtr node(new gazebo::transport::Node());
    node->Init();

    // Create ROS node and init
    ros::NodeHandle n;
    pub = n.advertise<std_msgs::Bool>("collision", 1000);

    // Listen to Gazebo contacts topic
    gazebo::transport::SubscriberPtr sub = node->Subscribe("/gazebo/default/physics/contacts", forcesCb);

    // Listen to ROS for position
    ros::Subscriber sub2 = n.subscribe("ground_truth/state", 1000, positionCb);

    // Busy wait loop...replace with your own code as needed.
    // Busy wait loop...replace with your own code as needed.
    while (true)
    {
        gazebo::common::Time::MSleep(20);

        // Spin ROS (needed for publisher) // (nope its actually for subscribers-calling callbacks ;-) )
        ros::spinOnce();


    // Mayke sure to shut everything down.

    }
    gazebo::client::shutdown();
}