//
// Created by deiwid on 19.2.10.
//

//#include "model_runner.h"
#include "c_api.h"

int main() {
    ModelRunner runner("/home/deiwid/VGTU/mag/g2p-rework/models/model-01_02_2019-214952-95.18-97.73-test");
    auto result = runner.infer({"l", "a", "b", "a", "s"}, 255);
    for(auto &c : result) {
        std::cout << c << " ";
    }
    std::cout << std::endl;


    void *run = getModelRunnerInstance("/home/deiwid/VGTU/mag/g2p-rework/models/model-01_02_2019-214952-95.18-97.73-test");
    const char* c[] = {"l", "a", "b", "a", "s"};
    const char** r = nullptr;
    size_t rn;
    modelRunnerInfer(run, c, 5, &r, &rn, 255);
    for(int i = 0; i < rn; i++) {
        std::cout << r[i] << " ";
        delete[] r[i];
    }
    delete[] r;
    std::cout << std::endl;
    deleteModelRunnerInstance(run);
    return 0;
}