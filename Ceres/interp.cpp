
#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/opencv.hpp>

#include "fmt/ranges.h"

#include "cnpy.h"

#include "ReprojErrInterp.hpp"
#include "config.h"
#include "h5_rw.hpp"
#include "utils.hpp"


int main(int argc, char** argv)
{
    auto [f_begin, f_end, tracks] = load_sift_tracks(DATA_DIR "sift-tracks.h5");

    fmt::print("Loaded {} tracks for frames {} to {}\n",
               tracks.size(), f_begin, f_end);

    std::map<int, std::array<double, 10>> kf_params;
    std::map<int, std::array<double, 4>> pt_avs;

    // clang-format off
    kf_params[f_begin] = {
        0, 0, 1, 0, // 4-vec axis-angle base pose
        0, 0, 1, 0, // 4-vec axis-angle angular velocity
        417, 0,     // focal length, focal length rate of change
    };
    // clang-format on

    for (size_t i = 0; i < tracks.size(); i++) {
        pt_avs[(int)i] = {0, 0, 1, 0};
    }

    ceres::Problem problem;

    ceres::Manifold* raxis_manifold = new ceres::SphereManifold<3>();

    for (int i = 0; i < (int)tracks.size(); i++) {
        auto t = tracks[i];
        for (int j = 1; j < (int)t.size(); j++) {
            auto* cost_fn = ReprojErrorInterp::create(t[j - 1], t[j], (double)(j - 1));
            problem.AddResidualBlock(cost_fn, nullptr,
                                     kf_params[f_begin].data() + 0,
                                     kf_params[f_begin].data() + 3,
                                     kf_params[f_begin].data() + 4,
                                     kf_params[f_begin].data() + 7,
                                     kf_params[f_begin].data() + 8,
                                     pt_avs[i].data() + 0,
                                     pt_avs[i].data() + 3);
        }
        problem.SetManifold(pt_avs[i].data() + 0, raxis_manifold);
        problem.SetParameterLowerBound(pt_avs[i].data() + 3, 0, 0.000);
        problem.SetParameterUpperBound(pt_avs[i].data() + 3, 0, 0.007);

        if (t[0].y() < 200) {
            problem.SetParameterBlockConstant(pt_avs[i].data() + 3);
        }
    }

    problem.SetManifold(kf_params[f_begin].data() + 0, raxis_manifold);
    problem.SetManifold(kf_params[f_begin].data() + 4, raxis_manifold);
    problem.SetParameterLowerBound(kf_params[f_begin].data() + 7, 0, 0.005);
    problem.SetParameterUpperBound(kf_params[f_begin].data() + 7, 0, 0.05);
    problem.SetParameterBlockConstant(kf_params[f_begin].data() + 8);

    ceres::Solver::Options options;
    options.max_num_iterations = 500;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    fmt::print("Optimized camera params:\n");

    auto& kfp = kf_params[f_begin];
    std::map<int, std::array<double, 4>> cam_params;

    for (int i = 0; i < f_end - f_begin; i++) {

        Eigen::Matrix3d R = utils::apply_angular_vel(
            {kfp[0], kfp[1], kfp[2], kfp[3]},
            {kfp[4], kfp[5], kfp[6], kfp[7]},
            static_cast<double>(i)
        );

        Eigen::Vector3d r;
        ceres::RotationMatrixToAngleAxis(R.data(), r.data());

        cam_params[i + f_begin] = {r[0], r[1], r[2],
                                   kfp[8] + kfp[9] * i};
    }

    for (auto& [id, params] : cam_params) {
        fmt::print("cam{}: {}\n", id, fmt::join(params, ", "));
    }

    double final_cost = summary.final_cost;
    final_cost /= static_cast<double>(summary.num_residual_blocks);
    final_cost = std::sqrt(final_cost);
    fmt::print("\nAverage reprojection error: {:.2f} pixels\n", final_cost);

    export_cam_params(DATA_DIR "optimized_cam_params.h5", cam_params);
    return 0;
}
