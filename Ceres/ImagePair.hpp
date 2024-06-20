
#pragma once

#include <array>
#include <vector>

struct ImagePair {
    double H[3][3];
    std::vector<std::array<double, 2>> src_pts;
    std::vector<std::array<double, 2>> dst_pts;
    int i;
    int j;
    bool still;
};
