//
// Created by deiwid on 19.2.10.
//

#ifndef MODEL_RUNNER_IO_MAP_H
#define MODEL_RUNNER_IO_MAP_H


#include <string>
#include <vector>
#include <map>
#include <tensorflow/core/framework/tensor.h>

namespace tf = tensorflow;

class IOMap {
private:
    std::vector<std::string> keys;
    std::map<std::string, uint> map;
public:
    IOMap() = default;
    IOMap(std::vector<std::string> keys);
    virtual ~IOMap() = default;
    size_t size();
    void encode(const std::vector<std::string> &list, tf::Tensor &tensor) const;
    std::vector<std::string> decode(const tf::Tensor &tensor) const;
};


#endif //MODEL_RUNNER_IO_MAP_H
