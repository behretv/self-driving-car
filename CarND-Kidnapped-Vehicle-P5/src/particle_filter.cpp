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
  if(isInitialized_){
    return;
  }

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
    particle.weight = 1.0;
    particles_.push_back(particle);
  }

  /* Add Gaussian distributed noise to x, y and theta */
  AddGaussianNoise(std);

  isInitialized_ = true;
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
      double theta0 = p.theta;
      p.x += velocity/yaw_rate* (sin(theta0 + yaw_rate * delta_t) - sin(theta0));
      p.y += velocity/yaw_rate* (cos(theta0) - cos(theta0 + yaw_rate * delta_t));
      p.theta += yaw_rate*delta_t;
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
  for(auto& o : observations){
    int map_id = -1;
    double min_distance = std::numeric_limits<double>::max();
    for(auto& p : predicted){
      double distance = dist(o.x, o.y, p.x, p.y);
      if (distance < min_distance)
      {
        min_distance = distance;
        map_id = p.id;
      }
    }
    o.id = map_id;
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

  for(auto& p : particles_){
    /* Temporrary variables */
    double theta = p.theta;
    double x_part = p.x;
    double y_part = p.y;

    /* Landmarks within sensor range */
    vector<LandmarkObs> valid_landmarks;
    for(auto& landmark : map_landmarks.landmark_list){
      float x_land = landmark.x_f;
      float y_land = landmark.y_f;
      if(dist(x_part, y_part, x_land, y_land) < sensor_range){
        valid_landmarks.push_back(LandmarkObs{landmark.id_i, x_land, y_land});
      }
    }

    /* Tranformation */
    vector<LandmarkObs> transformed_landmarks;
    for(auto& o : observations){
        double x_tran = x_part + (cos(theta) * o.x) - (sin(theta) * o.y);
        double y_tran = y_part + (sin(theta) * o.x) + (cos(theta) * o.y);
        transformed_landmarks.push_back(LandmarkObs{o.id, x_tran, y_tran});
    }

    /* Data association */
    dataAssociation(valid_landmarks, transformed_landmarks);

    /* Calculating Weights */
    p.weight = 1.0;
    double dx = 0;
    double dy = 0;
    for(auto& tl : transformed_landmarks){
      for(auto& vl : valid_landmarks){
        if (tl.id == vl.id)
        {
          dx = tl.x - vl.x;
          dy = tl.y - vl.y;
          break;
        }
      }
      double std_xy = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
      double std_xx = 1/(2*std_landmark[0]*std_landmark[0]);
      double std_yy = 1/(2*std_landmark[1]*std_landmark[1]);
      double weight = std_xy * exp( - (dx*dx*std_xx + dy*dy*std_yy));
      //p.weight *= (weight + 0.00001);
      p.weight = (weight == 0 ? 0.00001 : weight);
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  /* Extracting weights */
  double max_weights = std::numeric_limits<double>::min();
  std::vector<double> weights;
  for(auto& p : particles_){
    weights.push_back(p.weight);
    if(p.weight > max_weights){
      max_weights = p.weight;
    }
  }

  /* Discrete distribution */
  std::default_random_engine gen;
  std::uniform_int_distribution<int> dist_i(0, nParticles_);
  std::uniform_real_distribution<double> dist_d(0.0, max_weights);
  int idx = dist_i(gen);
  double weights_theshold = 0.0;

  std::vector<Particle> resampled_particles;
  unsigned int tmp_n_particles = nParticles_;
  while (tmp_n_particles)
  {
    weights_theshold += dist_d(gen) * 2;
    while(weights_theshold > weights[idx]){
      weights_theshold -= weights[idx];
      idx = (idx + 1) % nParticles_;
    }
    resampled_particles.push_back(particles_[idx]);
    tmp_n_particles--;
  }

  particles_ = resampled_particles;
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
