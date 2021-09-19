#!/bin/bash

USE_CPU=$1

git submodule update --init

pip install -r requirements.txt

pushd apex
git checkout ebcd7f084bba96bdb0c3fdf396c3c6b02e745042
if $USE_CPU; then
    pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
    git checkout 11faaca7c8ff7a7ba6d55854a9ee2689784f7ca5
    pip install -v --no-cache-dir --global-option="--cpp_ext" ./
else
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
fi
popd
