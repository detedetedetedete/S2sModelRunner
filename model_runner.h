//
// Created by deiwid on 19.2.5.
//

#ifndef MODEL_RUNNER_MODEL_RUNNER_H
#define MODEL_RUNNER_MODEL_RUNNER_H

#include <string>
#include <tensorflow/core/public/session.h>
#include "io_map.h"

namespace tf = tensorflow;

class ModelRunner {
private:
    std::unique_ptr<tf::Session> encoderSession;
    std::unique_ptr<tf::Session> decoderSession;

    std::vector<std::string> encoderInputs;
    std::vector<std::string> encoderOutputs;

    std::vector<std::string> decoderInputs;
    std::vector<std::string> decoderOutputs;

    std::vector<std::string> inputTokens;
    std::vector<std::string> outputTokens;

    IOMap inMap;
    IOMap outMap;

    int inputLength;
    std::string startToken = "[S]";
    std::string endToken = "[E]";
public:
    explicit ModelRunner(std::string model_path);
    virtual ~ModelRunner();
    std::vector<std::string> infer(const std::vector<std::string> &input, unsigned int max_len);
private:
    void loadModel(const std::string &model_path, std::unique_ptr<tf::Session> &session,
                   std::vector<std::string> &inputs, std::vector<std::string> &outputs) const;
};

#endif //MODEL_RUNNER_MODEL_RUNNER_H
