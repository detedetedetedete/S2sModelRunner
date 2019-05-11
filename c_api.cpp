//
// Created by deiwid on 19.2.11.
//

#include "c_api.h"

void* getModelRunnerInstance(const char *model_path) {
    return new ModelRunner(model_path);
}

void deleteModelRunnerInstance(void *ptr) {
    delete (ModelRunner*)ptr;
}

void modelRunnerInfer(void *ptr, const char **cinput, size_t cinput_n, const char ***cresult, size_t *cresult_n, size_t max_len) {
    std::vector<std::string> input;
    for(int i = 0; i < cinput_n; i++) {
        input.emplace_back(std::string(cinput[i]));
    }
    std::vector<std::string> result = ((ModelRunner*)ptr)->infer(input, max_len);
    char** _cresult = new char*[result.size()];
    *cresult_n = result.size();
    for(int i = 0; i < result.size(); i++) {
        size_t sz = result[i].size();
        _cresult[i] = new char[sz+1];
        memcpy(_cresult[i], result[i].c_str(), sz);
        _cresult[i][sz] = 0;
    }
    *cresult = (const char**)_cresult;
}
