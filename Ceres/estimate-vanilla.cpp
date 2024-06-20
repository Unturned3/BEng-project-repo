
#include <cstddef>
#include <iostream>
#include <map>
#include <opencv2/videoio.hpp>
#include <vector>
#include <filesystem>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/opencv.hpp>

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "cnpy.h"

#include "ImagePair.hpp"
#include "ReprojError.hpp"
#include "h5_rw.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;


int main(int argc, char** argv)
{
    if (argc < 6) {
        fmt::print("Usage: {} data_dir filename no_LC load_params use_stills\n", argv[0]);
        return 1;
    }

    fs::path data_dir = argv[1];
    fs::path file_name = argv[2];
    bool enable_LC = (std::string(argv[3]) == "yes");
    bool load_params = (std::string(argv[4]) == "yes");
    bool use_stills = (std::string(argv[5]) == "yes");

    std::string vid_stem = file_name.string().substr(0, 8);
    std::string out_path = data_dir / (vid_stem + "-opt-poses.h5");

    if (enable_LC) {
        fmt::print("Enabling loop closure constraints.\n");
    }

    auto [cam_indices, image_pairs] = load_image_pairs(data_dir / file_name);

    fmt::print("Optimizing poses for {}.\n", vid_stem);
    fmt::print("Loaded {} image pairs.\n", image_pairs.size());

    std::map<int, std::array<double, 4>> cam_params;

    if (load_params) {
        cam_params = import_cam_params(out_path);
    } else {
        for (int& i : cam_indices)
            cam_params[i] = {0, 0, 0, 640};
    }

    ceres::Problem problem;

    for (const ImagePair& p : image_pairs) {

        if (p.still && use_stills) {
            auto pts = utils::dummy_points();
            for (auto pt : pts) {
                ceres::CostFunction* cost_function =
                    RelativeReprojError::create(pt, pt);
                problem.AddResidualBlock(
                    cost_function,
                    //new ceres::ScaledLoss(nullptr, 100, ceres::TAKE_OWNERSHIP),
                    nullptr,
                    cam_params[p.i].data(),
                    cam_params[p.j].data());
            }
            continue;
        }

        for (size_t i = 0; i < p.src_pts.size(); i++) {
            ceres::CostFunction* cost_function =
                RelativeReprojError::create(p.src_pts[i], p.dst_pts[i]);

            if (p.i != p.j - 1) {
                if (!enable_LC) {
                    continue;
                }
                problem.AddResidualBlock(
                    cost_function,
                    //new ceres::ScaledLoss(nullptr, std::abs(p.j - p.i), ceres::TAKE_OWNERSHIP),
                    nullptr,
                    cam_params[p.i].data(),
                    cam_params[p.j].data()
                );
            } else {
                problem.AddResidualBlock(cost_function,
                                         //new ceres::CauchyLoss(0.5),
                                         new ceres::HuberLoss(1.0),
                                         //nullptr,
                                         cam_params[p.i].data(),
                                         cam_params[p.j].data());
            }

        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 500;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    //fmt::print("Optimized camera params:\n");
    //for (auto& v : cam_params) {
    //    auto [cam_id, params] = v;
    //    fmt::print("cam{}: {}\n", cam_id, fmt::join(params, ", "));
    //}

    double final_cost = summary.final_cost;
    final_cost /= static_cast<double>(summary.num_residual_blocks);
    final_cost = std::sqrt(final_cost);
    fmt::print("\nAverage reprojection error: {:.2f} pixels\n", final_cost);

    export_cam_params(out_path, cam_params);
    return 0;
}
