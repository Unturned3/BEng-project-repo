
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/iteration_callback.h>
#include <ceres/rotation.h>

#include "fmt/format.h"
#include "fmt/printf.h"
#include "fmt/ranges.h"

class LoggingCallback : public ceres::IterationCallback {
public:
    explicit LoggingCallback(bool log_to_stdout) : log_to_stdout_(log_to_stdout)
    {
    }

    ~LoggingCallback() {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
    {
        const char* kReportRowFormat =
            "% 4d: f:% 8e d:% 3.2e g:% 3.2e h:% 3.2e "
            "rho:% 3.2e mu:% 3.2e eta:% 3.2e li:% 3d";
        std::string output = fmt::sprintf(
            kReportRowFormat, summary.iteration, summary.cost,
            summary.cost_change, summary.gradient_max_norm, summary.step_norm,
            summary.relative_decrease, summary.trust_region_radius, summary.eta,
            summary.linear_solver_iterations);
        if (log_to_stdout_) {
            std::cout << output << std::endl;
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    const bool log_to_stdout_;
};
