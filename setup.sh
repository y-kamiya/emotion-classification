#!/bin/bash

USE_CPU=$1

git submodule update --init

pip install -r requirements.txt

pushd apex
git checkout ebcd7f084bba96bdb0c3fdf396c3c6b02e745042
if $USE_CPU; then
    pip install -v --no-cache-dir --global-option="--cpp_ext" ./
else
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
fi
popd
