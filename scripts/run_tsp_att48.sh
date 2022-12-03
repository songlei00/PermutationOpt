#!/bin/bash

seed_start=2022
seed_end=2023
epochs=10000
task=tsp
dataset=att48
task_name=tsp_$dataset
file_path=benchmarks/tsp_data/$dataset
dims=48

for ((seed=$seed_start; seed<=$seed_end; seed++))
do
    {
    python main.py \
        epochs=$epochs \
        task=$task \
        task.name=$task_name \
        task.file_path=$file_path \
        algorithm=mcts \
        algorithm.name=mcts_sa \
        algorithm.model.Cp=2000 \
        algorithm.model.max_propose=200 

    # sa
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=sa \
    #     algorithm.name=sa \
    #     seed=$seed

    # ea
    # python main.py \
    #     epochs=$(($epochs/20)) \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=ea \
    #     algorithm.name=ea \
    #     seed=$seed

    # dropout sa
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_sa \
    #     algorithm.model.inner_opt_type=sa \
    #     algorithm.model.n_init=1 \
    #     algorithm.model.active_dims=48 \
    #     algorithm.model.reset_freq=$epochs \
    #     +algorithm.model.decay=0.99 \
    #     +algorithm.model.T=100 \
    #     +algorithm.model.update_freq=100 \
    #     +algorithm.model.mutation_type=swap \
    #     seed=$seed

    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_sa \
    #     algorithm.model.inner_opt_type=sa \
    #     algorithm.model.n_init=1 \
    #     algorithm.model.active_dims=30 \
    #     algorithm.model.reset_freq=2000 \
    #     +algorithm.model.decay=0.99 \
    #     +algorithm.model.T=100 \
    #     +algorithm.model.update_freq=10 \
    #     +algorithm.model.mutation_type=swap \
    #     seed=$seed

    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_sa \
    #     algorithm.model.inner_opt_type=sa \
    #     algorithm.model.n_init=1 \
    #     algorithm.model.active_dims=15 \
    #     algorithm.model.reset_freq=2000 \
    #     +algorithm.model.decay=0.99 \
    #     +algorithm.model.T=100 \
    #     +algorithm.model.update_freq=10 \
    #     +algorithm.model.mutation_type=swap \
    #     seed=$seed

    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_sa \
    #     algorithm.model.inner_opt_type=sa \
    #     algorithm.model.n_init=1 \
    #     algorithm.model.active_dims=30 \
    #     algorithm.model.reset_freq=200 \
    #     +algorithm.model.decay=0.99 \
    #     +algorithm.model.T=100 \
    #     +algorithm.model.update_freq=10 \
    #     +algorithm.model.mutation_type=swap \
    #     seed=$seed

    # dropout ea
    # python main.py \
    #     epochs=$(($epochs/20)) \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_ea \
    #     algorithm.model.inner_opt_type=ea \
    #     algorithm.model.n_init=20 \
    #     algorithm.model.active_dims=48 \
    #     algorithm.model.reset_freq=$(($epochs/20)) \
    #     +algorithm.model.pop_size=20 \
    #     +algorithm.model.init_sampler_type=permutation \
    #     +algorithm.model.mutation_type=swap \
    #     +algorithm.model.crossover_type=order \
    #     seed=$seed

    # python main.py \
    #     epochs=$(($epochs/20)) \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_ea \
    #     algorithm.model.inner_opt_type=ea \
    #     algorithm.model.n_init=20 \
    #     algorithm.model.active_dims=30 \
    #     algorithm.model.reset_freq=$(($epochs/20)) \
    #     +algorithm.model.pop_size=20 \
    #     +algorithm.model.init_sampler_type=permutation \
    #     +algorithm.model.mutation_type=swap \
    #     +algorithm.model.crossover_type=order \
    #     seed=$seed

    # python main.py \
    #     epochs=$(($epochs/20)) \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_ea \
    #     algorithm.model.inner_opt_type=ea \
    #     algorithm.model.n_init=20 \
    #     algorithm.model.active_dims=15 \
    #     algorithm.model.reset_freq=$(($epochs/20)) \
    #     +algorithm.model.pop_size=20 \
    #     +algorithm.model.init_sampler_type=permutation \
    #     +algorithm.model.mutation_type=swap \
    #     +algorithm.model.crossover_type=order \
    #     seed=$seed

    # bops
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=bo \
    #     algorithm.name=bops \
    #     algorithm.model.active_dims=$dims \
    #     algorithm.model.acqf_opt_type=ea \
    #     seed=$seed

    # dropout bo
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=bo \
    #     algorithm.name=dropout_bo \
    #     algorithm.model.active_dims=10 \
    #     algorithm.model.acqf_opt_type=ea \
    #     algorithm.model.fillin_type=best_pos \
    #     seed=$seed

    # dropout random
    }
done