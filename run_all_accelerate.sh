#!/bin/bash

for i in 4 8
do
    for v in in_memory
    do
        accelerate launch --config_file config_${i}.yaml main.py --loader $v --n_gpus $i --parallel accelerator
    done
done
