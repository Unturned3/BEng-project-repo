
#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "utils.hpp"
struct ReprojErrorInterp {
    ReprojErrorInterp(const Eigen::Vector2d& src_pt,
                      const Eigen::Vector2d& observed_pt,
                      const double time)
        : src_pt(src_pt), observed_pt(observed_pt), time(time)
    {
    }

    template <typename T>
    bool operator()(const T* const kf_raxis,    // 3
                    const T* const kf_rmag,     // 1
                    const T* const kf_avaxis,   // 3
                    const T* const kf_avmag,    // 1
                    const T* const kf_focal,    // 2
                    const T* const pt_avaxis,   // 3
                    const T* const pt_avmag,    // 1
                    T* residuals) const
    {
        Eigen::Matrix3<T> R0 = utils::apply_angular_vel(
            {kf_raxis[0], kf_raxis[1], kf_raxis[2], kf_rmag[0]},
            {kf_avaxis[0], kf_avaxis[1], kf_avaxis[2], kf_avmag[0]},
            T(time)
        );

        Eigen::Matrix3<T> R1 = utils::apply_angular_vel(
            {kf_raxis[0], kf_raxis[1], kf_raxis[2], kf_rmag[0]},
            {kf_avaxis[0], kf_avaxis[1], kf_avaxis[2], kf_avmag[0]},
            T(time + 1)
        );

        /*  NOTE: pt_av is the inherent angular velocity of the point we
            are tracking. We do not multiply it by time, unlike we do for
            calculating R0 and R1. */
        Eigen::Vector4<T> av {pt_avaxis};
        av *= pt_avmag[0];
        Eigen::Matrix3<T> Rpt;
        ceres::AngleAxisToRotationMatrix(av.data(), Rpt.data());

        T f0 = kf_focal[0] + kf_focal[1] * T(time);
        T f1 = kf_focal[0] + kf_focal[1] * T(time + 1);

        T cx = T(320);
        T cy = T(240);

        Eigen::Vector3<T> src_pt_3d {
            (src_pt[0] - cx) / -f0,
            (src_pt[1] - cy) / f0,
            T(1),
        };

        Eigen::Vector3<T> dst_pt_3d = R1.transpose() * Rpt * R0 * src_pt_3d;

        Eigen::Vector2<T> dst_pt {
            (-f1 * dst_pt_3d[0]) / dst_pt_3d[2] + cx,
            (f1 * dst_pt_3d[1]) / dst_pt_3d[2] + cy,
        };

        residuals[0] = dst_pt[0] - T(observed_pt[0]);
        residuals[1] = dst_pt[1] - T(observed_pt[1]);

        return true;
    }

    static ceres::CostFunction* create(const Eigen::Vector2d& src_pt,
                                       const Eigen::Vector2d& observed_pt,
                                       const double time)
    {
        return new
            ceres::AutoDiffCostFunction<ReprojErrorInterp, 2, 3, 1, 3, 1, 2, 3, 1>(
            new ReprojErrorInterp(src_pt, observed_pt, time));
    }

private:
    const Eigen::Vector2d src_pt;
    const Eigen::Vector2d observed_pt;
    const double time;
};
