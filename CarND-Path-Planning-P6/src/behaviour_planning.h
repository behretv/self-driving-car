
#ifndef BEHAVIOUR_PLANNING_H
#define BEHAVIOUR_PLANNING_H

#include <vector>

using std::vector;

constexpr int kPathLength = 50;
constexpr double kDeltaTime = 0.02;
constexpr double kMaxSpeed = 49.5;
constexpr double kMeterPerSecondToMilesPerHour = 2.237;

class BehaviourPlanning
{
public:
    BehaviourPlanning(const double &ego_s, const double &ego_d, vector<vector<double>> &cars);
    ~BehaviourPlanning() = default;

    void Update(const double &ego_speed);

    int GetGoalLane() { return goal_lane_; }
    double GetAcceleration() { return acceleration_; }
    double GetSpeedCarAhead() { return speed_car_ahead_mph_; }

private:
    const double kMaxAcceleration = 0.2;
    const double kMinDistanceSameLane = 45;
    const double kMinDistanceNeighbourLaneForward = 50;
    const double kMinDistanceNeighbourLaneBackward = -15;

    bool is_traffic_ahead_ = false;
    bool is_traffic_left_lane_ = false;
    bool is_traffic_right_lane_ = false;

    int goal_lane_;
    double ego_s_;
    double ego_lane_;
    double acceleration_;
    double speed_car_ahead_mph_;
    vector<vector<double>> cars_;

    void CheckTrafficOnLanes();
    void ComputeGoalLane();
    void DetermineAcceleration(const double &speed);
    void SpeedCarAhead();
};

#endif //PATH_PLANNING_PATH_PLANNER_H
