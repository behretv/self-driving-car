#include "FusionEKF.h"
#include "Eigen/Dense"
#include "tools.h"
#include <iostream>
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  ekf_.P_ = MatrixXd(4, 4); // State covariance matrix P
  ekf_.F_ = MatrixXd(4, 4); // Tranistion matrix F
  ekf_.Q_ = MatrixXd(4, 4); // Process covariance matrix Q
  ekf_.x_ = VectorXd(4);    // State vector x

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0, 0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0, 0, 0.0009, 0, 0, 0, 0.09;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
  H_laser_ << 1, 0, 0, 0, // row 1
      0, 1, 0, 0;  // row 2

  VectorXd vec_ones = VectorXd(4);
  vec_ones << 1, 1, 1, 1;
  Hj_ << tools.CalculateJacobian(vec_ones);

  // The initial transition matrix F
  ekf_.F_ << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * You'll need to convert radar from polar to cartesian coordinates.
     */
    cout << "Kalman Filter Initialization:" << endl;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates
      //         and initialize state.
      float rho = measurement_pack.raw_measurements_[0];
      float theta = measurement_pack.raw_measurements_[1];
      float rho_dot = measurement_pack.raw_measurements_[2];
      ekf_.x_ << rho * cos(theta), // px
                 rho * sin(theta), // py
                 0,
                 0;
                 //rho_dot * cos(theta), // vx
                 //rho_dot * sin(theta); // vy
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // TODO: Initialize state.
      // set the state with the initial location and zero velocity
      ekf_.x_ << measurement_pack.raw_measurements_[0], // px
                 measurement_pack.raw_measurements_[1], // py
                 0,  // vx
                 0;  // vy
    } else {
      ekf_.x_ << 1, 1, 1, 1;
    }
    PrintStateVectorX();

    /**
     * TODO: Create the covariance matrix.
     */
    //ekf_.P_ << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1000;
    ekf_.P_ << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  /**
   * TODO: Update the state transition matrix F according to the new elapsed
   * time. Time is measured in seconds.
   */
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  /**
   * TODO: Update the process noise covariance matrix Q. 
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  // TODO: YOUR CODE HERE
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // set the acceleration noise components
  float noise_ax = 9.0;
  float noise_ay = 9.0;

  // set the process covariance matrix Q
  ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0, 0,
      dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay, dt_3 / 2 * noise_ax, 0,
      dt_2 * noise_ax, 0, 0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices R.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // TODO: Radar updates
    sensor_type_ = MeasurementPackage::RADAR;
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {
    // TODO: Laser updates
    sensor_type_ = MeasurementPackage::LASER;
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  PrintStateVectorX();
  //cout << "P_ = " << ekf_.P_ << endl;
}

void FusionEKF::PrintStateVectorX(){
  std::string types[2] = {"LASER", "RADAR"};
  cout << "x: [" 
       << ekf_.x_[0] << ", " 
       << ekf_.x_[1] << ", " 
       << ekf_.x_[2] << ", " 
       << ekf_.x_[3] << "] from " 
       << sensor_type_
       << endl;
}
