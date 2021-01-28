#include "behaviour_planning.h"
#include "helpers.h"

BehaviourPlanning::BehaviourPlanning(const double &ego_s,
                                     const double &ego_d,
                                     vector<vector<double>> &cars) : ego_s_(ego_s), cars_(cars)
{
    ego_lane_ = LateralDistance2CarLane(ego_d);
}

void BehaviourPlanning::CheckTrafficOnLanes()
{
    bool car_front = false;
    bool car_left = false;
    bool car_right = false;
    for (const auto &car : cars_)
    {
        double vx = car[0];
        double vy = car[1];
        double car_s = car[2];
        double car_d = car[3];

        int car_lane = LateralDistance2CarLane(car_d);

        if (car_lane == -1)
        {
            continue;
        }

        double car_v = sqrt(vx * vx + vy * vy);
        car_s += car_v * kPathLength * kDeltaTime;
        double distance = car_s - ego_s_;

        if (car_lane == ego_lane_)
        {
            car_front |= car_s > ego_s_ && distance < kMinDistanceSameLane;
        }
        else if (car_lane == ego_lane_ - 1)
        {
            car_left |= kMinDistanceNeighbourLaneBackward < distance && distance < kMinDistanceNeighbourLaneForward;
        }
        else if (car_lane == ego_lane_ + 1)
        {
            car_right |= kMinDistanceNeighbourLaneBackward < distance && distance < kMinDistanceNeighbourLaneForward;
        }
    }
    is_traffic_ahead_ = car_front;
    is_traffic_left_lane_ = car_left;
    is_traffic_right_lane_ = car_right;
}

void BehaviourPlanning::ComputeGoalLane()
{
    int goal_lane = 1;
    if (!is_traffic_ahead_)
    {
        if (ego_lane_ != 1)
        {
            if ((ego_lane_ == 0 && !is_traffic_right_lane_) || (ego_lane_ == 2 && !is_traffic_left_lane_))
            {
                goal_lane = 1;
            }
            else
            {
                goal_lane = ego_lane_;
            }
        }
    }
    else
    {
        if (!is_traffic_left_lane_ && ego_lane_ != 0)
        {
            goal_lane = ego_lane_ - 1;
        }
        else if (!is_traffic_right_lane_ && ego_lane_ != 2)
        {
            goal_lane = ego_lane_ + 1;
        }
        else
        {
            goal_lane = ego_lane_;
        }
    }

    if (goal_lane < 0)
    {
        goal_lane = 0;
    }
    else if (goal_lane > 2)
    {
        goal_lane = 2;
    }
    goal_lane_ = goal_lane;
}

void BehaviourPlanning::DetermineAcceleration(const double &speed)
{
    if (!is_traffic_ahead_)
    {
        acceleration_ = (speed < kMaxSpeed) ? kMaxAcceleration : 0;
    }
    else
    {
        acceleration_ = (speed > speed_car_ahead_mph_) ? -kMaxAcceleration : 0;
    }
}

void BehaviourPlanning::ComputeSpeedCarAhead()
{
    speed_car_ahead_mph_ = -1.0;
    for (const auto &car : cars_)
    {
        double vx = car[0];
        double vy = car[1];
        double car_s = car[2];
        double car_d = car[3];

        int car_lane = LateralDistance2CarLane(car_d);

        if (car_lane == -1)
        {
            continue;
        }

        double v = sqrt(vx * vx + vy * vy);
        car_s += v * kPathLength * kDeltaTime;

        if (car_lane == ego_lane_ && car_s > ego_s_)
        {
            speed_car_ahead_mph_ = v * kMeterPerSecondToMilesPerHour;
        }
    }
}
