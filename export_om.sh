#!/bin/bash

export TE_PARALLEL_COMPILER=1

atc --model=./outputs/edgead.onnx \
    --framework=5 \
    --output=./outputs/edgead \
    --soc_version=Ascend310P3 \
    --input_shape="image:1,3,256,256" 
