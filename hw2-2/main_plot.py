import random
import wandb
import argparse
import numpy as np
from tqdm import tqdm

from DP_solver_2_2_plot import (
    MonteCarloPolicyIteration,
    SARSA,
    Q_Learning,
)
from gridworld import GridWorld

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# STEP_REWARD       = -0.1
# GOAL_REWARD       = 1.0
# TRAP_REWARD       = -1.0
# DISCOUNT_FACTOR   = 0.99
# LEARNING_RATE     = 0.01
# EPSILON           = 0.2
# BUFFER_SIZE       = 10000
# UPDATE_FREQUENCY  = 200
# SAMPLE_BATCH_SIZE = 500


def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt"):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world

def run_MC_policy_iteration(grid_world: GridWorld, iter_num: int):
    print(bold(underline("MC Policy Iteration")))
    policy_iteration = MonteCarloPolicyIteration(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)

    # Record learning curve and loss curve
    learning_curves = []
    loss_curves = []
    for i in range(len(policy_iteration.episode_reward) - RUN_INTERVAL + 1):
        # Running average of 10 episodes
        learning_curves.append(np.mean(policy_iteration.episode_reward[i:i+RUN_INTERVAL]))
        loss_curves.append(np.mean(policy_iteration.episode_abs_est_loss[i:i+RUN_INTERVAL]))
        # print(f"Episode {i}-{i+9}: learn {learning_curves[-1]}, loss {loss_curves[-1]}")

    #     wandb.log({"learning_curve": learning_curves[-1]}, step=i)
    #     wandb.log({"loss_curve": loss_curves[-1]}, step=i)
    
    # wandb.finish()


    # grid_world.visualize(
    #     policy_iteration.get_max_state_values(),
    #     policy_iteration.get_policy_index(),
    #     title=f"MC Policy Iteration",
    #     show=False,
    #     filename=f"MC_policy_iteration_{iter_num}.png",
    # )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

    return learning_curves, loss_curves

def run_SARSA(grid_world: GridWorld, iter_num: int):
    print(bold(underline("SARSA Policy Iteration")))
    policy_iteration = SARSA(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)

    # Record learning curve and loss curve
    learning_curves = []
    loss_curves = []
    for i in range(len(policy_iteration.episode_reward) - RUN_INTERVAL + 1):
        # Running average of RUN_INTERVAL episodes
        learning_curves.append(np.mean(policy_iteration.episode_reward[i:i+RUN_INTERVAL]))
        loss_curves.append(np.mean(policy_iteration.episode_abs_est_loss[i:i+RUN_INTERVAL]))

    # grid_world.visualize(
    #     policy_iteration.get_max_state_values(),
    #     policy_iteration.get_policy_index(),
    #     title=f"SARSA",
    #     show=False,
    #     filename=f"SARSA_iteration_{iter_num}.png",
    # )
    # history = grid_world.run_policy(policy_iteration.get_policy_index())
    # print(f"Solved in {bold(green(len(history)))} steps")
    # print(history)
    # print(
    #     f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    # )
    # grid_world.reset()
    # print()

    return learning_curves, loss_curves

def run_Q_Learning(grid_world: GridWorld, iter_num: int):
    print(bold(underline("Q_Learning Policy Iteration")))
    policy_iteration = Q_Learning(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            buffer_size=BUFFER_SIZE,
            update_frequency=UPDATE_FREQUENCY,
            sample_batch_size=SAMPLE_BATCH_SIZE,
            )
    policy_iteration.run(max_episode=iter_num)

    # Record learning curve and loss curve
    learning_curves = []
    loss_curves = []
    for i in range(len(policy_iteration.episode_reward) - RUN_INTERVAL + 1):
        # Running average of RUN_INTERVAL episodes
        learning_curves.append(np.mean(policy_iteration.episode_reward[i:i+RUN_INTERVAL]))
        loss_curves.append(np.mean(policy_iteration.episode_abs_est_loss[i:i+RUN_INTERVAL]))

        # if i % (RUN_INTERVAL * 100) == 0:
        #     print(f"Episode {i}-{i+RUN_INTERVAL-1}: learn {learning_curves[-1]}, loss {loss_curves[-1]}")


    # grid_world.visualize(
    #     policy_iteration.get_max_state_values(),
    #     policy_iteration.get_policy_index(),
    #     title=f"Q_Learning",
    #     show=False,
    #     filename=f"Q_Learning_iteration_{iter_num}.png",
    # )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

    return learning_curves, loss_curves


def get_args():
    parser = argparse.ArgumentParser()
    # WandB
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--experiment_name", type=str)

    # Grid World
    # STEP_REWARD       = -0.1
    # GOAL_REWARD       = 1.0
    # TRAP_REWARD       = -1.0
    # DISCOUNT_FACTOR   = 0.99
    # LEARNING_RATE     = 0.01
    # EPSILON           = 0.2
    # BUFFER_SIZE       = 10000
    # UPDATE_FREQUENCY  = 200
    # SAMPLE_BATCH_SIZE = 500
    parser.add_argument("--step_reward", type=float, default=-0.1)
    parser.add_argument("--goal_reward", type=float, default=1.0)
    parser.add_argument("--trap_reward", type=float, default=-1.0)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--update_frequency", type=int, default=200)
    parser.add_argument("--sample_batch_size", type=int, default=500)

    # Experiment
    parser.add_argument("--algorithm", type=str)
    parser.add_argument("--run_interval", type=int, default=10)
    parser.add_argument("--max_episode", type=int, default=512000)

    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args


if __name__ == "__main__":
    args = get_args()
    
    # Set hyperparameters of grid world
    STEP_REWARD       = args.step_reward
    GOAL_REWARD       = args.goal_reward
    TRAP_REWARD       = args.trap_reward
    DISCOUNT_FACTOR   = args.discount_factor
    LEARNING_RATE     = args.learning_rate
    EPSILON           = args.epsilon
    BUFFER_SIZE       = args.buffer_size
    UPDATE_FREQUENCY  = args.update_frequency
    SAMPLE_BATCH_SIZE = args.sample_batch_size
    RUN_INTERVAL      = args.run_interval

    # WandB – Initialize a new run
    wandb.init(project=args.project_name,
               name=args.experiment_name)
    config = wandb.config
    config.step_reward = STEP_REWARD
    config.goal_reward = GOAL_REWARD
    config.trap_reward = TRAP_REWARD
    config.discount_factor = DISCOUNT_FACTOR
    config.learning_rate = LEARNING_RATE
    config.epsilon = EPSILON
    config.buffer_size = BUFFER_SIZE
    config.update_frequency = UPDATE_FREQUENCY
    config.sample_batch_size = SAMPLE_BATCH_SIZE

    # Initialize grid world
    grid_world = init_grid_world()
    # run_MC_policy_iteration(grid_world, 512000)
    # run_SARSA(grid_world, 512000)
    # run_Q_Learning(grid_world, 50000)

    # pbar = tqdm(total=4)
    # learnings, losses = [], []
    # for EPSILON in [0.1, 0.2, 0.3, 0.4]:
    #     # config.epsilon = EPSILON
    #     print(f"EPSILON: {EPSILON}")
    #     learn, loss = run_MC_policy_iteration(grid_world, args.max_episode)
    #     learnings.append(learn)
    #     losses.append(loss)
    #     pbar.update(1)

    if args.algorithm == "MC":
        learnings, losses = run_MC_policy_iteration(grid_world, args.max_episode)
    elif args.algorithm == "SARSA":
        learnings, losses = run_SARSA(grid_world, args.max_episode)
    elif args.algorithm == "Q-Learning":
        learnings, losses = run_Q_Learning(grid_world, args.max_episode)
    else:
        raise ValueError("Invalid algorithm name")

    # Plot on WandB
    print("Plotting on WandB ...")
    # wandb.log({"MC_learning": wandb.plot.line_series(
    #                    xs=[i + args.run_interval - 1 for i in range(len(learnings[0]))],
    #                    ys=learnings,
    #                    keys=["eps 0.1", "eps 0.2", "eps 0.3", "eps 0.4"],
    #                    title="Learning curves",
    #                    xname="# episode")})
    # wandb.log({"MC_loss": wandb.plot.line_series(
    #                       xs=[i + args.run_interval - 1 for i in range(len(losses[0]))],
    #                       ys=losses,
    #                       keys=["eps 0.1", "eps 0.2", "eps 0.3", "eps 0.4"],
    #                       title="Loss curves",
    #                       xname="# episode")})
    # wandb.finish()

    # Only one curve in each plot
    pbar = tqdm(total=len(learnings))
    for i, (learn, loss) in enumerate(zip(learnings, losses)):
        wandb.log({"learning_curve": learn}, step=i)
        wandb.log({"loss_curve": loss}, step=i)
        pbar.update(1)
    
    wandb.finish()



