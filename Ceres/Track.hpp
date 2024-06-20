
#pragma once

#include <Eigen/Dense>
#include <vector>

struct Track {
    int uid;
    int start_frame_idx;
    std::vector<Eigen::Vector2d> pts;
};
