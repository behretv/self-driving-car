#include "gtest/gtest.h"
#include "../src/particle_filter.h"

TEST(blaTest, test1) {
    //arrange
    //act
    //assert
    Particle p;
    p.x = 1.0;
    ParticleFilter pf = ParticleFilter();
    EXPECT_EQ(pf.getAssociations(p),  "");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
