#!/bin/bash

seed_start=2022
seed_end=2024
epochs=200

for ((seed=$seed_start; seed<=$seed_end; seed++))
do
    {
    # qap
    # python main.py \
    #     epochs=$epochs \
    #     task=qap \
    #     algorithm.model.active_dims=15 \
    #     algorithm.model.acqf_opt_type=ls \
    #     algorithm.model.device=cpu \
    #     seed=$seed

    # # tsp
    # python main.py \
    #     epochs=$epochs \
    #     task=tsp \
    #     task.file_path=benchmarks/tsp_data/att48 \
    #     algorithm=bo \
    #     algorithm.model.active_dims=48 \
    #     algorithm.model.acqf_opt_type=ea \
    #     algorithm.model.device=cpu \
    #     seed=$seed

    python main.py \
        epochs=$epochs \
        task=tsp \
        task.name=tsp_lin105 \
        task.file_path=benchmarks/tsp_data/lin105 \
        algorithm=bo \
        algorithm.model.active_dims=105 \
        algorithm.model.acqf_opt_type=ea \
        seed=$seed
    }
done