#!/bin/bash

for i in 1 2 4 8
do
    for v in ir_datasets indexed in_memory
    do
        python main.py --loader $v --n_gpus $i
    done
done
