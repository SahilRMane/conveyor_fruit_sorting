// Calculate position reference offsets for the fixed inspection poses.
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: pose_bias SCENE POSES");
        char error[2048] = {};
        std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(mj_loadXML(argv[1], nullptr, error, sizeof(error)), mj_deleteModel);
        if (!model) throw std::runtime_error(error);
        std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model.get()), mj_deleteData);
        const auto poses = YAML::LoadFile(argv[2])["views"];
        const int home = mj_name2id(model.get(), mjOBJ_KEY, "home");
        if (home < 0) throw std::runtime_error("Missing home keyframe");
        std::cout << std::setprecision(17);
        for (const auto* name : {"assess", "inspect_left", "inspect_right"}) {
            mj_resetDataKeyframe(model.get(), data.get(), home);
            for (int j = 0; j < 7; ++j) {
                const auto joint = "panda_joint" + std::to_string(j + 1);
                const int id = mj_name2id(model.get(), mjOBJ_JOINT, joint.c_str());
                if (id < 0 || model->jnt_qposadr[id] != j || model->jnt_dofadr[id] != j ||
                    mj_name2id(model.get(), mjOBJ_ACTUATOR, joint.c_str()) != j)
                    throw std::runtime_error("Panda joint or actuator order differs");
                data->qpos[j] = poses[name]["joints"][j].as<double>();
            }
            mj_forward(model.get(), data.get());
            std::cout << name;
            for (int j = 0; j < 7; ++j) {
                const double kp = model->actuator_gainprm[j * mjNGAIN];
                if (kp <= 0 || std::abs(model->actuator_biasprm[j * mjNBIAS + 1] + kp) > 1e-6)
                    throw std::runtime_error("Expected Panda position servos");
                const double offset = data->qfrc_bias[j] / kp;
                const double target = data->qpos[j] + offset;
                if (!std::isfinite(offset) || std::abs(offset) > .02 ||
                    target < model->actuator_ctrlrange[2 * j] || target > model->actuator_ctrlrange[2 * j + 1])
                    throw std::runtime_error("Pose offset exceeds its bound");
                std::cout << ' ' << offset;
            }
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
