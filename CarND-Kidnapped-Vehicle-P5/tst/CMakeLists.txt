
# ---- Tests ---
# Checks if GTest has been installed as sharde libarry before, if not do so or skip the tests.
# How to install as shared library:
# sudo apt-get install libgtest-dev
# cd /usr/src/gtest
# sudo cmake CMakeLists.txt
# sudo make
# sudo cp *.a /usr/lib
find_package(GTest REQUIRED)

if(GTest_FOUND)
    include_directories(${GTEST_INCLUDE_DIRS})

    # Tests need to be added as executables first
    add_executable(particle_filter_tests
        test.cpp
        particle_filter_test.cpp
        )

    # Should be linked to the main library, as well as the Catch2 testing library
    target_link_libraries(particle_filter_tests PRIVATE ${GTEST_LIBRARIES} pthread particle_filter_lib)

    # If you register a test, then ctest and make test will run it.
    # You can also run examples and check the output, as well.
    add_test(NAME testlibtest COMMAND particle_filter_tests) # Command can be a target

endif(GTest_FOUND)
