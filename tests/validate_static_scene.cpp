// Check contacts and free-body settling with the installed MuJoCo library.
#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  char error[2048] = {};
  mjModel* m = mj_loadXML(argv[1], nullptr, error, sizeof(error));
  if (!m) { std::fprintf(stderr, "%s\n", error); return 1; }
  mjData* d = mj_makeData(m);
  try {
    require(m->nq == 16 && m->nv == 15 && m->nu == 8, "Unexpected model dimensions");
    int tray = mj_name2id(m, mjOBJ_BODY, "tray");
    int support = mj_name2id(m, mjOBJ_BODY, "stationary_support");
    int robot = mj_name2id(m, mjOBJ_BODY, "link0");
    int joint = mj_name2id(m, mjOBJ_JOINT, "tray_free_joint");
    int home = mj_name2id(m, mjOBJ_KEY, "home");
    require(tray >= 0 && support >= 0 && robot >= 0 && joint >= 0 && home >= 0, "Missing scene names");
    require(m->jnt_type[joint] == mjJNT_FREE, "Tray must have a free joint");
    require(m->body_jntnum[support] == 0, "Support must be fixed");
    require(std::fabs(m->body_mass[tray] - 0.52) < 1e-9, "Unexpected tray mass");
    int qa = m->jnt_qposadr[joint], va = m->jnt_dofadr[joint];
    require(qa == 9, "Panda joints must precede the tray free joint");
    auto is_robot = [&](int body) {
      while (body > 0) {
        if (body == robot) return true;
        body = m->body_parentid[body];
      }
      return false;
    };
    for (int run = 0; run < 2; ++run) {
      if (run == 0) {
        mj_resetDataKeyframe(m, d, home);
      } else {
        // Match ROS startup: initialize only robot coordinates and controls.
        mj_resetData(m, d);
        for (int i = 0; i < 9; ++i) d->qpos[i] = m->key_qpos[home*m->nq+i];
        for (int i = 0; i < 8; ++i) d->ctrl[i] = m->key_ctrl[home*m->nu+i];
      }
      mj_forward(m, d);
      require(std::fabs(d->qpos[qa+2] - 0.307) < 1e-9, "Wrong initial tray height");
      double initial[3] = {d->qpos[qa], d->qpos[qa+1], d->qpos[qa+2]};
      while (d->time < 5.0) {
        mj_step(m, d);
        for (int i = 0; i < d->ncon; ++i) {
          int a = m->geom_bodyid[d->contact[i].geom[0]];
          int b = m->geom_bodyid[d->contact[i].geom[1]];
          bool env_a = a == tray || a == support;
          bool env_b = b == tray || b == support;
          require(!((env_a && is_robot(b)) || (env_b && is_robot(a))), "Robot touches tray/support at home");
        }
      }
      mj_forward(m, d);
      double normal_force = 0;
      int count = 0;
      for (int i = 0; i < d->ncon; ++i) {
        int a = m->geom_bodyid[d->contact[i].geom[0]];
        int b = m->geom_bodyid[d->contact[i].geom[1]];
        if ((a == tray && b == support) || (a == support && b == tray)) {
          mjtNum force[6]; mj_contactForce(m, d, i, force);
          normal_force += force[0]; ++count;
        }
      }
      double speed = 0;
      for (int i = 0; i < m->nq; ++i) require(std::isfinite(d->qpos[i]), "Non-finite position");
      for (int i = 0; i < 6; ++i) speed = std::fmax(speed, std::fabs(d->qvel[va+i]));
      double drift = std::hypot(d->qpos[qa]-initial[0], d->qpos[qa+1]-initial[1]);
      require(count > 0 && normal_force > 4.5 && normal_force < 5.7, "Support force does not balance tray weight");
      require(std::fabs(d->qpos[qa+2]-0.306) < 0.001, "Tray does not rest on support");
      require(drift < 0.001 && speed < 0.001, "Tray does not settle");
      require(std::fabs(d->qpos[qa+3]) > 0.999, "Tray tilts");
      for (int i = 0; i < mjNWARNING; ++i) require(d->warning[i].number == 0, "MuJoCo physics warning");
      std::printf("%s: nq=%ld nv=%ld nu=%ld mass=%.3f kg z=%.6f m contacts=%d normal_force=%.6f N drift=%.9f m speed=%.9f\n",
                  run == 0 ? "Home keyframe" : "ROS-style startup", static_cast<long>(m->nq), static_cast<long>(m->nv), static_cast<long>(m->nu),
                  m->body_mass[tray], d->qpos[qa+2], count, normal_force, drift, speed);
    }
    std::printf("PASS: stationary support; free tray; rigid handle; stable contact; home clearance\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    mj_deleteData(d); mj_deleteModel(m); return 1;
  }
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
