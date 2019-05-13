#! /bin/bash

mkdir -p ./extern
git clone https://github.com/nlohmann/json ./extern/json
cd ./extern/json
git checkout db53bdac1926d1baebcb459b685dcd2e4608c355
cd ../..
git clone https://github.com/tensorflow/tensorflow ./extern/tensorflow
cd ./extern/tensorflow
git checkout a6d8ffae097d0132989ae4688d224121ec6d8f35
cd ../..

