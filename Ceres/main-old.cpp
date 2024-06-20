
#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/opencv.hpp>

#include "fmt/format.h"
#include "fmt/printf.h"
#include "fmt/ranges.h"

#include "cnpy.h"

#include "ImagePair.hpp"
#include "ReprojError.hpp"
#include "config.h"
#include "h5_rw.hpp"
#include "utils.hpp"

/*
void tracks_to_residuals(const std::vector<Track>& tracks,
                         std::map<int, std::array<double, 4>>& cam_params,
                         std::map<int, std::array<double, 3>>& pt_motions,
                         ceres::Problem& problem, double reg)
{
    for (auto& t : tracks) {
        size_t f = static_cast<size_t>(t.start_frame_idx);
        for (size_t i = 0; i < t.pts.size() - 1; i++) {
            auto cost_fn =
                // RelativeReprojErrorWithMotion::create(t.pts[i], t.pts[i + 1],
                // reg);
                RelativeReprojError::create(t.pts[i], t.pts[i + 1]);

            problem.AddResidualBlock(cost_fn, nullptr,
                                     cam_params[int(f + i)].data(),
                                     cam_params[int(f + i + 1)].data());
            // pt_motions[int(t.uid)].data());
        }
    }
}
*/

int main(int argc, char** argv)
{
    auto [cam_indices, image_pairs] = load_image_pairs(DATA_DIR "pairs.h5");
    // auto [track_uids, tracks] = load_tracks(DATA_DIR "tracks.h5");

    std::map<int, std::array<double, 4>> cam_params;
    // for (int& i : cam_indices) {
    //     cam_params[i] = {0, 0, 0, 640};
    // }
    for (int i = 0; i < 900; i++) {
        cam_params[i] = {0, 0, 0, 640};
    }

    std::map<int, std::array<double, 3>> pt_motions;
    // for (int& i : track_uids) {
    //     pt_motions[i] = {0, 0, 0};
    // }

    // fmt::print("Number of tracks: {}\n", int(tracks.size()));

    ceres::Problem problem;

    for (const ImagePair& p : image_pairs) {
        for (size_t i = 0; i < p.src_pts.size(); i++) {
            ceres::CostFunction* cost_function =
                RelativeReprojError::create(p.src_pts[i], p.dst_pts[i]);
            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5),
                                     cam_params[p.i].data(),
                                     cam_params[p.j].data());
        }
    }

    // tracks_to_residuals(tracks, cam_params, pt_motions, problem,
    //                     std::stod(argv[1]));

    // for (int& i : track_uids) {
    //     problem.SetParameterBlockConstant(pt_motions[i].data());
    // }

    ceres::Solver::Options options;
    options.max_num_iterations = 500;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    fmt::print("Optimized camera params:\n");

    for (auto& v : cam_params) {
        auto [cam_id, params] = v;
        fmt::print("cam{}: {}\n", cam_id, fmt::join(params, ", "));
    }

    double final_cost = summary.final_cost;
    final_cost /= static_cast<double>(summary.num_residual_blocks);
    final_cost = std::sqrt(final_cost);
    fmt::print("\nAverage reprojection error: {:.2f} pixels\n", final_cost);

    export_cam_params(DATA_DIR "optimized_cam_params.h5", cam_params);
    return 0;
}
