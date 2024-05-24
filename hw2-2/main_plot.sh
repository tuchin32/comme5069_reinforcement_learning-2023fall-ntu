#!/bin/bash

if [ "$1" == "MC" ]
then
    # MC policy iteration
    echo "MC policy iteration"
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "MC_eps_0.1" --algorithm "MC" --epsilon 0.1
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "MC_eps_0.2" --algorithm "MC" --epsilon 0.2
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "MC_eps_0.3" --algorithm "MC" --epsilon 0.3
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "MC_eps_0.4" --algorithm "MC" --epsilon 0.4

elif [ "$1" == "SARSA" ]
then
    # SARSA
    echo "SARSA"
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "SARSA_eps_0.1" --algorithm "SARSA" --epsilon 0.1
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "SARSA_eps_0.2" --algorithm "SARSA" --epsilon 0.2
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "SARSA_eps_0.3" --algorithm "SARSA" --epsilon 0.3
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "SARSA_eps_0.4" --algorithm "SARSA" --epsilon 0.4

elif [ "$1" == "Q-Learning" ]
then
    # Q-Learning
    echo "Q-Learning"
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_eps_0.1" --algorithm "Q-Learning" --epsilon 0.1
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_eps_0.2" --algorithm "Q-Learning" --epsilon 0.2
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_eps_0.3" --algorithm "Q-Learning" --epsilon 0.3
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_eps_0.4" --algorithm "Q-Learning" --epsilon 0.4

else
    # # discount factor: {0.5, 0.9, 0.99}, 0.99 is the default value
    # echo "discount factor: {0.5, 0.9, 0.99}"
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "MC_df_0.5" --algorithm "MC" --discount_factor 0.5
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "MC_df_0.9" --algorithm "MC" --discount_factor 0.9
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "MC_df_0.99" --algorithm "MC" --discount_factor 0.99
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "SARSA_df_0.5" --algorithm "SARSA" --discount_factor 0.5
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "SARSA_df_0.9" --algorithm "SARSA" --discount_factor 0.9
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "SARSA_df_0.99" --algorithm "SARSA" --discount_factor 0.99
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_df_0.5" --algorithm "Q-Learning" --discount_factor 0.5 --max_episode 50000
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_df_0.9" --algorithm "Q-Learning" --discount_factor 0.9 --max_episode 50000
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_df_0.99" --algorithm "Q-Learning" --discount_factor 0.99 --max_episode 50000 

    # # learning rate: {0.1, 0.01, 0.001}, 0.01 is the default value
    # echo "learning rate: {0.1, 0.01, 0.001}"
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "MC_lr_0.1" --algorithm "MC" --learning_rate 0.1
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "MC_lr_0.01" --algorithm "MC" --learning_rate 0.01
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "MC_lr_0.001" --algorithm "MC" --learning_rate 0.001
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "SARSA_lr_0.1" --algorithm "SARSA" --learning_rate 0.1
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "SARSA_lr_0.01" --algorithm "SARSA" --learning_rate 0.01
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "SARSA_lr_0.001" --algorithm "SARSA" --learning_rate 0.001
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_lr_0.1" --algorithm "Q-Learning" --learning_rate 0.1 --max_episode 50000
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_lr_0.01" --algorithm "Q-Learning" --learning_rate 0.01 --max_episode 50000
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_lr_0.001" --algorithm "Q-Learning" --learning_rate 0.001 --max_episode 50000

    # # update frequency: {30, 100, 200, 500}, 200 is the default value
    # echo "update frequency: {30, 100, 200, 500}"
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_uf_30" --algorithm "Q-Learning" --update_frequency 30 --max_episode 50000
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_uf_100" --algorithm "Q-Learning" --update_frequency 100 --max_episode 50000
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_uf_200" --algorithm "Q-Learning" --update_frequency 200 --max_episode 50000
    # python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_uf_500" --algorithm "Q-Learning" --update_frequency 500 --max_episode 50000

    # sample_batch_size: {100, 500, 2000, 10000}, 500 is the default value
    echo "sample_batch_size: {100, 500, 2000, 10000}"
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_sbs_100" --algorithm "Q-Learning" --sample_batch_size 100 --max_episode 50000
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_sbs_500" --algorithm "Q-Learning" --sample_batch_size 500 --max_episode 50000
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_sbs_2000" --algorithm "Q-Learning" --sample_batch_size 2000 --max_episode 50000
    python main_plot.py --project_name "rl_hw2-2" --experiment_name "Q-Learning_sbs_10000" --algorithm "Q-Learning" --sample_batch_size 10000 --max_episode 50000
fi