#include "PID.h"
#include <iostream>

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double kp, double ki, double kd)
{
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  kp_ = kp;
  ki_ = ki;
  kd_ = kd;

  i_error_ = 0;
}

void PID::UpdateError(double cte)
{
  /**
   * TODO: Update PID errors based on cte.
   */
  double previous_cte = p_error_;
  p_error_ = cte;
  i_error_ += cte;
  d_error_ = cte - previous_cte;
}

double PID::TotalError()
{
  /**
   * TODO: Calculate and return the total error
   */
  return -(kp_ * p_error_ + ki_ * i_error_ + kd_ * d_error_); // TODO: Add your total error calc here!
}

void PID::SetParameterByIndex(double value, int index)
{
  switch (index)
  {
  case 0:
    kp_ += value;
    break;
  case 1:
    ki_ += value;
    break;
  case 2:
    kd_ += value;
    break;
  default:
    std::cout << "Warning: Trying to set k-parameter with invalid index:" << index << std::endl;
    break;
  }
}

void PID::PrintK()
{
  std::cout << "P=" << kp_ << "|" << ki_ << "|" << kd_ << std::endl;
}
