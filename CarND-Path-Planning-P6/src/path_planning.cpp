#include <iostream>
#include <math.h>

#include "behaviour_planning.h"
#include "helpers.h"
#include "path_planning.h"

using std::vector;

PathPlanning::PathPlanning(vector<double> &map_waypoints_x,
                           vector<double> &map_waypoints_y,
                           vector<double> &map_waypoints_s) : map_waypoints_x_(map_waypoints_x),
                                                              map_waypoints_y_(map_waypoints_y),
                                                              map_waypoints_s_(map_waypoints_s)
{
}

void PathPlanning::ComputeTrajectory(const double &acceleration, const double &speed_car_ahead)
{
    double rel_x = 0;
    double rel_y = 0;

    // Compute the next 2-3 values of the trajectory
    for (size_t i = 0; i < kPathLength - next_vec_x_.size(); i++)
    {
        // Increase speed until maximum
        if (ego_v_ < kMaxSpeed)
        {
            ego_v_ += acceleration;
        }
        else if (ego_v_ > kMaxSpeed)
        {
            ego_v_ = kMaxSpeed;
        }
        else if (ego_v_ > speed_car_ahead && speed_car_ahead > 0)
        {
            ego_v_ += acceleration;
        }

        // Compute step size along the spline
        double step_size_x = ComputeStepSize();

        // Relative x/y-values
        rel_x = rel_x + step_size_x;
        rel_y = spline_(rel_x);

        // Absolut x/y-values
        double next_x = ego_x_ + (rel_x * cos(ego_yaw_) - rel_y * sin(ego_yaw_));
        double next_y = ego_y_ + (rel_x * sin(ego_yaw_) + rel_y * cos(ego_yaw_));

        next_vec_x_.push_back(next_x);
        next_vec_y_.push_back(next_y);
    }
}

double PathPlanning::ComputeStepSize()
{
    double goal_x = waypoint_vec_s_[0]; // default 30
    double goal_y = spline_(goal_x);
    double goal_distance = sqrt(pow(goal_x, 2) + pow(goal_y, 2));
    double v_in_ms = ego_v_ / kMeterPerSecondToMilesPerHour;
    double num_steps = goal_distance / (kDeltaTime * v_in_ms);
    double step_size_x = goal_x / num_steps;
    return step_size_x;
}

void PathPlanning::SetDefaultStartPointsForSplines(double car_x, double car_y, double car_yaw)
{
    /*
     * Set default start points, will be overwritten in SetStartPointsForSpline if prev_path has
     * more than 2 points
     */
    ego_x_ = car_x;
    ego_y_ = car_y;
    ego_yaw_ = deg2rad(car_yaw);
    ego_prev_x_ = car_x - cos(car_yaw);
    ego_prev_y_ = car_y - sin(car_yaw);
}

void PathPlanning::SetStartPointsForTrajectory(vector<vector<double>> &prev_path)
{
    if (prev_path_size_ > 0)
    {
        next_vec_x_.clear();
        next_vec_y_.clear();
        next_vec_x_ = prev_path[0];
        next_vec_y_ = prev_path[1];
    }
}

void PathPlanning::SetStartPointsForSpline(vector<vector<double>> &prev_path)
{
    prev_path_size_ = prev_path.empty() ? 0 : prev_path[0].size();

    // If no previous path
    if (prev_path_size_ >= 2)
    {
        // Use last 2 values of previous path
        ego_x_ = prev_path[0][prev_path_size_ - 1];
        ego_y_ = prev_path[1][prev_path_size_ - 1];
        ego_prev_x_ = prev_path[0][prev_path_size_ - 2];
        ego_prev_y_ = prev_path[1][prev_path_size_ - 2];

        ego_yaw_ = atan2(ego_y_ - ego_prev_y_, ego_x_ - ego_prev_x_);
    }

    spline_vec_x_.clear();
    spline_vec_y_.clear();
    spline_vec_x_.push_back(ego_prev_x_);
    spline_vec_y_.push_back(ego_prev_y_);
    spline_vec_x_.push_back(ego_x_);
    spline_vec_y_.push_back(ego_y_);
}

void PathPlanning::ComputeSpline(double car_s, int goal_lane)
{
    // Set way points
    int goal_d = 2 + 4 * goal_lane;
    for (auto s : waypoint_vec_s_)
    {
        auto waypoint = getXY(car_s + s, goal_d, map_waypoints_s_, map_waypoints_x_, map_waypoints_y_);
        spline_vec_x_.push_back(waypoint[0]);
        spline_vec_y_.push_back(waypoint[1]);
    }

    // Transform from world to car coordinates
    for (size_t i = 0; i < spline_vec_x_.size(); i++)
    {
        double shift_x = spline_vec_x_[i] - ego_x_;
        double shift_y = spline_vec_y_[i] - ego_y_;

        spline_vec_x_[i] = shift_x * cos(-ego_yaw_) - shift_y * sin(-ego_yaw_);
        spline_vec_y_[i] = shift_x * sin(-ego_yaw_) + shift_y * cos(-ego_yaw_);
    }

    spline_.set_points(spline_vec_x_, spline_vec_y_);
}
