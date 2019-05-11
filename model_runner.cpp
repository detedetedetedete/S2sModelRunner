#include "model_runner.h"

#include <fstream>

#include <absl/strings/match.h>

#include <nlohmann/json.hpp>

namespace nl = nlohmann;

ModelRunner::ModelRunner(std::string model_path) {
    std::string encoder_path = model_path + "/encoder_inference_model.pb";
    std::string decoder_path = model_path + "/decoder_inference_model.pb";
    loadModel(encoder_path, encoderSession, encoderInputs, encoderOutputs);
    loadModel(decoder_path, decoderSession, decoderInputs, decoderOutputs);

    std::ifstream in(model_path + "/model.json");
    nl::json json;
    in >> json;

    inputTokens = json["in_tokens"].get<std::vector<std::string>>();
    outputTokens = json["out_tokens"].get<std::vector<std::string>>();
    inputLength = json["max_in_length"];
    outputTokens.push_back(startToken);
    outputTokens.push_back(endToken);

    inMap = IOMap(inputTokens);
    outMap = IOMap(outputTokens);
}

ModelRunner::~ModelRunner() = default;

void ModelRunner::loadModel(const std::string &model_path, std::unique_ptr<tf::Session> &session,
        std::vector<std::string> &inputs, std::vector<std::string> &outputs) const {
    tf::GraphDef graph_def;
    tf::Status load_status = tf::ReadBinaryProto(tf::Env::Default(), model_path, &graph_def);

    if(!load_status.ok()) {
        std::cerr << "Could not load " << model_path << std::endl;
        exit(1);
    }

    session.reset(tf::NewSession(tf::SessionOptions()));
    tf::Status session_create_status = session->Create(graph_def);

    if(!load_status.ok()) {
        std::cerr << "Could not create session for model " << model_path << std::endl;
        exit(2);
    }

    for(int i = 0; i < graph_def.node_size(); i++) {
        auto &node = graph_def.node(i);
        if(absl::EndsWith(node.name(), "_input")) {
            inputs.push_back(node.name());
        } else if(absl::EndsWith(node.name(), "_output")) {
            outputs.push_back(node.name());
        }
    }
}

std::vector<std::string> ModelRunner::infer(const std::vector<std::string> &input, unsigned int max_len) {
    std::vector<std::string> result;
    std::vector<tf::Tensor> decoder_inputs;
    decoder_inputs.emplace_back(tf::Tensor(tf::DataType::DT_FLOAT, {1, 1, (long long) outMap.size()}));
    outMap.encode({startToken}, decoder_inputs[0]);
    result.push_back(startToken);
    {
        std::vector<tf::Tensor> encoder_outputs;
        tf::Tensor encoder_input(tf::DataType::DT_FLOAT, {1, inputLength, (long long) inMap.size()});
        inMap.encode(input, encoder_input);
        tf::Status encoder_run_status = encoderSession->Run(
                {{encoderInputs[0], encoder_input}}, encoderOutputs, {}, &encoder_outputs);
        if (!encoder_run_status.ok()) {
            std::cerr << "Failed to run encoderSession->Run: " << encoder_run_status.error_message() << std::endl;
            exit(4);
        }
        std::move(encoder_outputs.begin(), encoder_outputs.end(), std::back_inserter(decoder_inputs));
    }

    while(outMap.decode(decoder_inputs[0])[0] != endToken && result.size() < max_len) {
        std::vector<std::pair<std::string, tf::Tensor>> decoder_input;
        for(int i = 0; i < decoderInputs.size(); i++) {
            decoder_input.emplace_back(std::pair<std::string, tf::Tensor>(decoderInputs[i], decoder_inputs[i]));
        }
        std::vector<tf::Tensor> decoder_outputs;
        tf::Status decoder_run_status = decoderSession->Run(
                decoder_input, decoderOutputs, {}, &decoder_outputs);
        if (!decoder_run_status.ok()) {
            std::cerr << "Failed to run decoderSession->Run: " << decoder_run_status.error_message() << std::endl;
            exit(5);
        }
        result.emplace_back(outMap.decode(decoder_outputs[0])[0]);
        decoder_inputs = decoder_outputs;

    }
    return result;
}
