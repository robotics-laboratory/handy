#include "params.h"

bool handy::CameraIntrinsicParameters::save(const std::string path_to_yaml_file) const {
    std::ofstream param_file(path_to_yaml_file);
    if (!param_file) {
        return false;
    }

    YAML::Emitter output_yaml;
    output_yaml << YAML::BeginMap;

    output_yaml << YAML::Key << "camera_matrix";
    output_yaml << YAML::Value << YAML::BeginSeq;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            output_yaml << camera_matrix(i, j);
        }
    }
    output_yaml << YAML::EndSeq;

    output_yaml << YAML::Key << "distorsion_coefs";
    output_yaml << YAML::Value << YAML::BeginSeq;
    for (int i = 0; i < 5; ++i) {
        output_yaml << dist_coefs[i];
    }
    output_yaml << YAML::EndSeq;

    output_yaml << YAML::EndMap;

    param_file << output_yaml.c_str();
    param_file.close();
    return true;
}