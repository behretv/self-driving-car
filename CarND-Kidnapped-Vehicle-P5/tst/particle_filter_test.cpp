#include "gtest/gtest.h"
#include "particle_filter.h"

TEST(blaTest, test1) {
    //arrange
    //act
    //assert
    Particle p;
    p.x = 1.0;
    ParticleFilter pf = ParticleFilter();
    EXPECT_EQ(pf.getAssociations(p),  "");
}
