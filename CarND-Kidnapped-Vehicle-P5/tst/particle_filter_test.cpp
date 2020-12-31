#include "gtest/gtest.h"
#include "../src/particle_filter.h"

void Check50PercentPercentil(const std::vector<double>& samples, const double& mu){
    unsigned int n_higher = 0;
    unsigned int n_lower = 0;
    for (auto& s : samples)
    {
        if (s < mu)
        {
            n_lower++;
        }
        else
        {
            n_higher++;
        }
    }
    EXPECT_NEAR(n_lower, n_higher, samples.size()/10+1);

}


TEST(GetAssociationsTest, TestPositiveVectors) {
    //arrange
    //act
    //assert
    Particle p;
    p.associations = {1, 2, 3};
    auto pf = ParticleFilter();
    EXPECT_EQ(pf.getAssociations(p),  "1 2 3");
}

TEST(InitTest, TestNumberOfParticles){
    ParticleFilter filter = ParticleFilter();
    double std_pos[3] = {1.0, 2.0, 3.0};
    filter.init(0.0, 0.0, 0.0, std_pos);
    EXPECT_EQ(filter.particles.size(), 100);
}

TEST(InitTest, TestGaussianDistribution){
    ParticleFilter filter = ParticleFilter();
    double mu_x = 0.0;
    double mu_y = 0.0;
    double mu_theta = 0.0;
    double std_pos[3] = {1.0, 2.0, 3.0};
    filter.init(mu_x, mu_y, mu_theta, std_pos);

    /* Approximatly 50% should be higher and 50% lower than the mean*/
    std::vector<double> samples_x;
    std::vector<double> samples_y;
    std::vector<double> samples_theta;
    for(auto& particle : filter.particles){
        samples_x.push_back(particle.x);
        samples_y.push_back(particle.y);
        samples_theta.push_back(particle.theta);
    }
    Check50PercentPercentil(samples_x, mu_x);
    Check50PercentPercentil(samples_y, mu_y);
    Check50PercentPercentil(samples_theta, mu_theta);
}
