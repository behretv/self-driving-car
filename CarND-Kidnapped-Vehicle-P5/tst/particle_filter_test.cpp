#include "gtest/gtest.h"
#include "../src/particle_filter.h"

TEST(GetAssociations_Test, test1) {
    //arrange
    //act
    //assert
    Particle p;
    p.associations = {1, 2, 3};
    ParticleFilter pf = ParticleFilter();
    EXPECT_EQ(pf.getAssociations(p),  "1 2 3");
}
