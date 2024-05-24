import numpy as np
import json
from collections import defaultdict

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)

    def get_all_state_values(self) -> np.array:
        return self.values


class MonteCarloPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float):
        """Constructor for MonteCarloPrediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with first-visit Monte-Carlo method

        # Initialize values and returns
        values = self.get_all_state_values()
        returns = [[] for _ in range(self.state_space)]
        episode = []

        current_state = self.grid_world.reset()
        while self.grid_world.check():
            # Generate an episode following the policy: (state(t-1), reward(t))
            next_state, reward, done = self.grid_world.step()
            episode.append((current_state, reward))
            current_state = next_state

            if done:
                G = 0
                # Loop for each step of episode, t = T-1, T-2, ..., 0
                reversed_episode = list(reversed(episode))
                for i, (state, reward) in enumerate(reversed_episode):
                    G = self.discount_factor * G + reward

                    # Update the value unless the pair (state, reward) appears in the episode
                    if state not in [x[0] for x in reversed_episode[i+1:]]:
                        returns[state].append(G)
                        values[state] = np.mean(returns[state])

                episode = []
                # current_state = self.grid_world.reset()

        self.values = values


class TDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with TD(0) Algorithm
        values = self.get_all_state_values()

        current_state = self.grid_world.reset()
        while self.grid_world.check():
            next_state, reward, done = self.grid_world.step()
            # print('curr', current_state)

            td_target = reward + self.discount_factor * values[next_state] * (1 - done)
            td_error = td_target - values[current_state]
            values[current_state] += self.lr * td_error
            current_state = next_state

            if done:
                pass
                # current_state = self.grid_world.reset()
            
        self.values = values


class NstepTDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, num_step: int):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with N-step TD Algorithm
        values = self.get_all_state_values()
        t = 0
        T = np.inf
        rewards, states = [], []

        current_state = self.grid_world.reset()
        states.append(current_state)
        while self.grid_world.check():
            if t < T:
                next_state, reward, done = self.grid_world.step()
                rewards.append(reward)
                states.append(next_state)
                current_state = next_state
                if done:
                    T = t + 1
            
            tau = t - self.n + 1

            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + self.n, T) + 1):
                    G += self.discount_factor**(i - tau - 1) * rewards[i - 1]
                if tau + self.n < T:
                    # print('tau', tau, 'n', self.n, 't', t, 'rewards', len(rewards))
                    G += self.discount_factor**self.n * values[states[tau + self.n]]
                values[states[tau]] += self.lr * (G - values[states[tau]])

            t += 1

            if tau == T - 1:
                rewards, states = [], []
                # current_state = self.grid_world.reset()
                states.append(current_state)
                t = 0
                T = np.inf



        self.values = values