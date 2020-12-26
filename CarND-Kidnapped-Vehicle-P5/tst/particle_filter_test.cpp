#include "gtest/gtest.h"
#include "../src/particle_filter.h"

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
    EXPECT_EQ(filter.particles.size(), 1);
}
