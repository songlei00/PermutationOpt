#!/bin/bash

seed_start=2022
seed_end=2022
epochs=50000
task=qap
task_name=qap

for ((seed=$seed_start; seed<=$seed_end; seed++))
do
    {
    # sa
    python main.py \
        epochs=$epochs \
        task=$task \
        task.name=$task_name \
        algorithm=sa \
        algorithm.name=sa \
        seed=$seed

    # # ea
    python main.py \
        epochs=$(($epochs/20)) \
        task=$task \
        task.name=$task_name \
        algorithm=ea \
        algorithm.name=ea \
        seed=$seed

    # dropout sa
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_sa \
    #     algorithm.model.inner_opt_type=sa \
    #     algorithm.model.active_dims=10 \
    #     algorithm.model.reset_freq=1 \
    #     +algorithm.model.decay=0.99 \
    #     +algorithm.model.T=100 \
    #     +algorithm.model.update_freq=100 \
    #     +algorithm.model.mutation_type=swap \
    #     seed=$seed

    # dropout ea
    # python main.py \
    #     epochs=$(($epochs/20)) \
    #     task=$task \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_ea \
    #     algorithm.model.inner_opt_type=ea \
    #     algorithm.model.active_dims=10 \
    #     algorithm.model.reset_freq=20 \
    #     +algorithm.model.pop_size=20 \
    #     +algorithm.model.init_sampler_type=permutation \
    #     +algorithm.model.mutation_type=swap \
    #     +algorithm.model.crossover_type=order \
    #     seed=$seed
    }
done