/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"
#include "helper_functions.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "./helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  nParticles_ = 100;

  /*
   * Set the number of particles. Initialize all particles to
   * first position (based on estimates of x, y, theta and their uncertainties
   * from GPS) and all weights to 1.
  */
  for(int i = 0; i < nParticles_; i++){
    auto particle = Particle();
    particle.id = i;
    particle.x = x;
    particle.y = y;
    particle.theta = theta;
    particle.weight = 1;
    particles_.push_back(particle);
  }

  /* Add Gaussian distributed noise to x, y and theta */
  AddGaussianNoise(std);
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
    for(auto& p : particles_){
      double x0 = p.x;
      double y0 = p.y;
      double theta0 = p.theta;
      p.x = x0 + velocity/yaw_rate* (sin(theta0 + yaw_rate * delta_t) - sin(theta0));
      p.y = y0 + velocity/yaw_rate* (cos(theta0) - cos(theta0 + yaw_rate * delta_t));
      p.theta = theta0 + yaw_rate*delta_t;
    }

    /* Add Gaussian distributed noise to x, y and theta */
    AddGaussianNoise(std_pos);
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  // transform to map x coordinate
  for(auto& p : predicted){
    double min_distance = 999.9;
    for(auto& o : observations){
      double distance = dist(o.x, o.y, p.x, p.y);
      if (distance < min_distance)
      {
        min_distance = distance;
      }

    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  /* Tranformation */
  for(auto& o :observations){
      double theta = 0;
      double x_part = 0.0;
      double y_part = 0.0;
      double x_obs = o.x;
      double y_obs = o.y;
      o.x = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
      o.y = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
}

void ParticleFilter::SetAssociations(Particle* particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle->associations = associations;
  particle->sense_x = sense_x;
  particle->sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

void ParticleFilter::AddGaussianNoise(double std[]){
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  std::normal_distribution<double> dist_x(0, std_x);
  std::normal_distribution<double> dist_y(0, std_y);
  std::normal_distribution<double> dist_theta(0, std_theta);

  /*
   * Add random Gaussian noise to each particle.
  */
  std::default_random_engine gen;
  for(auto& p : particles_){
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }
}
