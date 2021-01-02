/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define EPS 0.00001

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]){
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  if (isInitialized_){
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

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // Calculate new state.
  for (auto& particle : particles_) {

  	double theta = particle.theta;

    if ( fabs(yaw_rate) < EPS ) {
      particle.x += velocity * delta_t * cos( theta );
      particle.y += velocity * delta_t * sin( theta );
    } else {
      particle.x += velocity / yaw_rate * ( sin( theta + yaw_rate * delta_t ) - sin( theta ) );
      particle.y += velocity / yaw_rate * ( cos( theta ) - cos( theta + yaw_rate * delta_t ) );
      particle.theta += yaw_rate * delta_t;
    }

  }
    // Adding noise.
    AddGaussianNoise(std_pos);
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
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
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  double std_xy = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
  double std_xx = 1/(2*std_landmark[0]*std_landmark[0]);
  double std_yy = 1/(2*std_landmark[1]*std_landmark[1]);

  for (int i = 0; i < nParticles_; i++) {
    /* Temporrary variables */
    Particle& p = particles_[i];
    double theta = p.theta;
    double x = p.x;
    double y = p.y;

    /* Landmarks within sensor range */
    vector<LandmarkObs> valid_landmarks;
    for(auto& landmark : map_landmarks.landmark_list){
      float x_land = landmark.x_f;
      float y_land = landmark.y_f;
      if(dist(x, y, x_land, y_land) < sensor_range){
        valid_landmarks.push_back(LandmarkObs{landmark.id_i, x_land, y_land});
      }
    }

    /* Tranformation */
    vector<LandmarkObs> transformed_landmarks;
    for(auto& o : observations){
        double x_tran = x + (cos(theta) * o.x) - (sin(theta) * o.y);
        double y_tran = y + (sin(theta) * o.x) + (cos(theta) * o.y);
        transformed_landmarks.push_back(LandmarkObs{o.id, x_tran, y_tran});
    }

    // Observation association to landmark.
    dataAssociation(valid_landmarks, transformed_landmarks);

    /* Calculating Weights */
    p.weight = 1.0;
    double weight = 1.0;
    for(auto& tl : transformed_landmarks){
      for(auto& vl : valid_landmarks){
        if (tl.id == vl.id)
        {
          double dx = tl.x - vl.x;
          double dy = tl.y - vl.y;
          weight = std_xy * exp( - (dx*dx*std_xx + dy*dy*std_yy));
          break;
        }
      }
      p.weight *= (weight == 0 ? 0.00001 : weight);
    }
  }
}

void ParticleFilter::resample() {
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
  std::default_random_engine gen_;
  std::uniform_int_distribution<int> dist_i(0, nParticles_);
  std::uniform_real_distribution<double> dist_d(0.0, max_weights);
  int idx = dist_i(gen_);
  double weights_theshold = 0.0;

  std::vector<Particle> resampled_particles;
  unsigned int tmp_n_particles = nParticles_;
  while (tmp_n_particles)
  {
    weights_theshold += dist_d(gen_) * 2;
    while(weights_theshold > weights[idx]){
      weights_theshold -= weights[idx];
      idx = (idx + 1) % nParticles_;
    }
    resampled_particles.push_back(particles_[idx]);
    tmp_n_particles--;
  }

  particles_ = resampled_particles;}

void ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
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
  for(auto& p : particles_){
    p.x += dist_x(gen_);
    p.y += dist_y(gen_);
    p.theta += dist_theta(gen_);
  }
}
