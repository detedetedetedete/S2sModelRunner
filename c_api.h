//
// Created by deiwid on 19.2.11.
//

#ifndef MODEL_RUNNER_C_API_H
#define MODEL_RUNNER_C_API_H

#include "model_runner.h"

extern "C" {
    void* getModelRunnerInstance(const char *model_path);
    void deleteModelRunnerInstance(void *ptr);
    void modelRunnerInfer(void *ptr, const char **cinput, size_t cinput_n, const char ***cresult, size_t *cresult_n, size_t max_len);
};

#endif //MODEL_RUNNER_C_API_H
