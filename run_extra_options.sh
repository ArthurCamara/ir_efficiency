#!/bin/bash

# First, number of workers
for n in 0 1 2 4 8
do
    python main.py --loader ir_datasets --n_gpus 4 --num_workers $n
done

# Baseline
# Does pinning memory make any difference?
python main.py --loader ir_datasets --n_gpus 4 --pin_memory

# Finally, RamDisk. Does it help?
python main.py --loader ir_datasets --n_gpus 4 --ramdisk