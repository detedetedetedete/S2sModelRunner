#include <utility>

//
// Created by deiwid on 19.2.10.
//

#include "io_map.h"

IOMap::IOMap(std::vector<std::string> keys) {
    this->keys = std::move(keys);
    for(unsigned int i = 0; i < this->keys.size(); i++) {
        this->map[this->keys[i]] = i;
    }
}

void IOMap::encode(const std::vector<std::string> &list, tf::Tensor &tensor) const {
    tf::TTypes<float>::Flat data = tensor.flat<float>();
    std::fill(data.data(), data.data() + data.size(), 0.);
    unsigned long sz = keys.size();
    for(unsigned int i = 0; i < list.size(); i++) {
        std::map<std::string, unsigned int>::const_iterator it;
        if((it = map.find(list[i])) == map.end()) {
            std::cerr << "Token '" << list[i] << "' not in map." << std::endl;
            exit(3);
        }

        data(sz*i + it->second) = 1.;
    }
}

std::vector<std::string> IOMap::decode(const tf::Tensor &tensor) const {
    std::vector<std::string> result;
    unsigned long sz = keys.size();
    tf::TTypes<float>::ConstFlat data = tensor.flat<float>();
    long data_sz = data.size();
    unsigned long inner_idx = 0;
    unsigned long idx = 0;

    float max = data(0);
    unsigned long max_idx = 0;
    while(idx != data_sz) {
        float val = data(idx);
        if(val > max) {
            max = val;
            max_idx = idx;
        }

        idx++;
        inner_idx++;
        if(inner_idx == sz) {
            inner_idx = 0;
            result.push_back(keys[max_idx % sz]);

            if(idx != data_sz) {
                max = data(idx);
                max_idx = idx;
            }
        }
    }

    return result;
}

size_t IOMap::size() {
    return keys.size();
}
