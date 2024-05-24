import numpy as np
from collections import deque
from gridworld import GridWorld
# from tqdm import tqdm


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
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space 
        self.policy_index = np.zeros(self.state_space, dtype=int)

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        
        # raise NotImplementedError
        # print('state length', len(state_trace))
        # print('action length', len(action_trace))
        # print('reward length', len(reward_trace))
        # import ipdb; ipdb.set_trace()

        # Monte Carlo policy evaluation
        # Calculate the return G_t for each state
        G = 0
        for i in reversed(range(len(state_trace))):
            # print(i)
            G = self.discount_factor * G + reward_trace[i]
            s = state_trace[i]
            a = action_trace[i]
            self.q_values[s, a] += self.lr * (G - self.q_values[s, a])
        
        

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy

        # raise NotImplementedError

        # epsilon-greedy improvement
        # pi(a|s) = 1 - epsilon + epsilon / |A(s)|, if a = argmax_a Q(s,a)
        #         = epsilon / |A(s)|,               otherwise
        for state in range(self.state_space):
            best_action = self.q_values[state].argmax()
            self.policy[state] = self.epsilon / self.action_space
            self.policy[state][best_action] += (1 - self.epsilon)



    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        # pbar = tqdm(total=max_episode)

        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace   = [current_state]
        action_trace  = []
        reward_trace  = []
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            
            # raise NotImplementedError
            action = np.random.choice(self.action_space, p=self.policy[current_state])
            next_state, reward, done = self.grid_world.step(action)
            action_trace.append(action)
            reward_trace.append(reward)
            # print('s a r:', current_state, action, reward)
            current_state = next_state

            if done:
                self.policy_evaluation(state_trace, action_trace, reward_trace)
                self.policy_improvement()
                state_trace  = [current_state]
                action_trace = []
                reward_trace = []
                iter_episode += 1

                # pbar.update(1)
                # if iter_episode % 10000 == 0:
                #     print(f"Episode {iter_episode} finished.")

            else:
                state_trace.append(current_state)

class SARSA(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        
        # raise NotImplementedError
        # Evaluate Q value
        td_error = r + self.discount_factor * self.q_values[s2, a2] * (1 - is_done) - self.q_values[s, a]
        self.q_values[s, a] += self.lr * td_error

        # Improve the policy
        best_action = self.q_values[s].argmax()
        self.policy[s] = self.epsilon / self.action_space
        self.policy[s][best_action] += (1 - self.epsilon)


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            # raise NotImplementedError
            prev_s = current_state if prev_s is None else prev_s
            prev_a = np.random.choice(self.action_space, p=self.policy[prev_s])

            while not is_done:
                next_s, prev_r, is_done = self.grid_world.step(prev_a)
                next_a = np.random.choice(self.action_space, p=self.policy[next_s])
                self.policy_eval_improve(prev_s, prev_a, prev_r, next_s, next_a, is_done)
                
                prev_s = next_s
                prev_a = next_a

            iter_episode += 1
            is_done = False

            # if iter_episode % 10000 == 0:
            #     print(f"Episode {iter_episode} finished.")



class Q_Learning(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        # raise NotImplementedError
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        # raise NotImplementedError
        return np.random.choice(len(self.buffer), self.sample_batch_size)

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        # raise NotImplementedError
        
        # Evaluate Q value
        td_error = r + self.discount_factor * self.q_values[s2].max() * (1 - is_done) - self.q_values[s, a]
        self.q_values[s, a] += self.lr * td_error

        # Improve the policy
        best_action = self.q_values[s].argmax()
        self.policy[s] = self.epsilon / self.action_space
        self.policy[s][best_action] += (1 - self.epsilon)


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        transition_count = 0
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            # raise NotImplementedError

            prev_s = current_state if prev_s is None else prev_s
            is_done = False

            while not is_done:
                prev_a = np.random.choice(self.action_space, p=self.policy[prev_s])
                next_s, prev_r, is_done = self.grid_world.step(prev_a)
                self.add_buffer(prev_s, prev_a, prev_r, next_s, is_done)
                transition_count += 1

                if transition_count % self.update_frequency == 0:
                    # batch = self.sample_batch()
                    # self.policy_eval_improve(batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4])
                    batches_index = self.sample_batch()
                    batches = [self.buffer[i] for i in batches_index]
                
                    for batch in batches:
                        self.policy_eval_improve(batch[0], batch[1], batch[2], batch[3], batch[4])
                
                prev_s = next_s

            iter_episode += 1
            is_done = False

            # if iter_episode % 10000 == 0:
            #     print(f"Episode {iter_episode} finished.")