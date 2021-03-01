
#ifndef UKF_FILTER_H
#define UKF_FILTER_H

#include <rclcpp/time.hpp>
#include <cinematography_msgs/msg/multi_do_farray.hpp>
#include <cinematography_msgs/msg/multi_dof.hpp>

void ukf_init();

cinematography_msgs::msg::MultiDOF get_state(rclcpp::Duration duration);

//cinematography_msgs::msg::MultiDOFarray build_forecast(int points, rclcpp::Duration point_duration);

std::vector<cinematography_msgs::msg::MultiDOF> ukf_iterate(rclcpp::Duration point_duration, int forecast_length, float vx, float vy, float vz, float wx, float wy, float wz);

void ukf_meas_clear();

void ukf_set_bb(float x, float y, float area);

void ukf_set_position(float x, float y, float z);

void ukf_set_yaw(float yaw);

void ukf_set_pitch(float pitch);

void ukf_set_hde(float hde);

#endif