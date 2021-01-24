#ifndef PATH_PLANNING_PATH_PLANNER_H
#define PATH_PLANNING_PATH_PLANNER_H

#include "spline.h"
#include <memory>

using std::vector;

class PathPlanning
{
public:
    // Constructor
    PathPlanning(vector<double> &map_waypoints_x, vector<double> &map_waypoints_y, vector<double> &map_waypoints_s);
    ~PathPlanning() = default;

    vector<double> GetNextVectorX() { return next_vec_x_; }
    vector<double> GetNextVectorY() { return next_vec_y_; }

    void ComputeTrajectory(const double &speed_inc, const double &speed_car_ahead);
    void SetDefaultStartPointsForSplines(double car_x, double car_y, double car_yaw);
    void SetStartPointsForSpline(vector<vector<double>> &prev_path);
    void ComputeSpline(double car_s, int goal_lane);
    void SetStartPointsForTrajectory(vector<vector<double>> &prev_path);

private:
    const vector<double> waypoint_vec_s_ = {50.0, 70.0, 90.0};

    // Number of previous path points
    size_t prev_path_size_ = 0;

    // Ego vehicle values
    double ego_x_;
    double ego_y_;
    double ego_yaw_;
    double ego_v_ = 0.0;
    double ego_prev_x_;
    double ego_prev_y_;

    // Map waypoints
    vector<double> map_waypoints_x_;
    vector<double> map_waypoints_y_;
    vector<double> map_waypoints_s_;

    vector<double> spline_vec_x_;
    vector<double> spline_vec_y_;

    vector<double> next_vec_x_;
    vector<double> next_vec_y_;

    tk::spline spline_;

    double ComputeStepSize();
};

#endif //PATH_PLANNING_PATH_PLANNER_H
