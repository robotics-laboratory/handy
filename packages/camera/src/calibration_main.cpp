#include "calibration.h"

#include <yaml-cpp/yaml.h>

int main(int argc, char *argv[]) {
    const YAML::Node param_node = YAML::LoadFile("calibration.yaml")["parameters"];
    handy::calibration::CalibrationNode node(param_node);

    return 0;
}
