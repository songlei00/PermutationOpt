#!/bin/bash

seed_start=2022
seed_end=2022
epochs=15

for ((seed=$seed_start; seed<=$seed_end; seed++))
do
    {
    # qap
    python main.py \
        epochs=$epochs \
        task=qap \
        algorithm.model.active_dims=15 \
        algorithm.model.acqf_opt_type=ls \
        seed=$seed

    # tsp
    python main.py \
        epochs=$epochs \
        task=tsp \
        task.file_path=benchmarks/tsp_data/att48 \
        algorithm=bo \
        algorithm.model.active_dims=48 \
        algorithm.model.acqf_opt_type=ls \
        seed=$seed

    python main.py \
        epochs=$epochs \
        task=tsp \
        task.file_path=benchmarks/tsp_data/bayg29 \
        algorithm=bo \
        algorithm.model.active_dims=29 \
        algorithm.model.acqf_opt_type=ls \
        seed=$seed
    }
done