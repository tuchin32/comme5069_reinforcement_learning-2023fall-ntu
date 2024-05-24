import numpy as np

from gridworld import GridWorld
from queue import PriorityQueue


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
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        # raise NotImplementedError
        next_state, reward, done = self.grid_world.step(state, action)
        q_value = reward + self.discount_factor * self.values[next_state] * (1 - done)
        return q_value        


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        # raise NotImplementedError
        new_state_value = 0.0
        for action in range(self.grid_world.get_action_space()):
            new_state_value += self.policy[state, action] * self.get_q_value(state, action)
        return new_state_value

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        # raise NotImplementedError
        new_values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state)
        
        delta = np.max(np.abs(self.values - new_values))
        self.values = new_values

        return delta

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        # raise NotImplementedError
        delta = self.threshold + 1
        while delta >= self.threshold:
            delta = self.evaluate()


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        # raise NotImplementedError
        action = self.policy[state]
        new_state_value = self.get_q_value(state, action)
        return new_state_value

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        # raise NotImplementedError
        new_values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state)
        
        delta = np.max(np.abs(self.values - new_values))
        self.values = new_values

        return delta

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        # raise NotImplementedError
        new_policy = np.zeros(self.grid_world.get_state_space(), dtype=int)
        for state in range(self.grid_world.get_state_space()):
            new_policy[state] = np.argmax([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])

        policy_stable = np.all(self.policy == new_policy)
        self.policy = new_policy

        return policy_stable

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        # raise NotImplementedError
        policy_stable = False
        while not policy_stable:
            delta = self.threshold + 1
            while delta >= self.threshold:
                delta = self.policy_evaluation()
            policy_stable = self.policy_improvement()


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        # raise NotImplementedError
        new_state_value = np.max([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
        return new_state_value

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        # raise NotImplementedError
        new_values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state)

        delta = np.max(np.abs(self.values - new_values))
        self.values = new_values

        return delta

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        # raise NotImplementedError
        new_policy = np.zeros(self.grid_world.get_state_space(), dtype=int)
        for state in range(self.grid_world.get_state_space()):
            new_policy[state] = np.argmax([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])

        self.policy = new_policy

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        # raise NotImplementedError
        delta = self.threshold + 1
        while delta >= self.threshold:
            delta = self.policy_evaluation()
        
        self.policy_improvement()


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)
    
    
    def inplace_dynamic_programming(self):
        # Policy Evaluation
        delta = self.threshold + 1
        while delta >= self.threshold:
            old_values = self.values.copy()

            state_candidates = np.arange(self.grid_world.get_state_space())
            for state in state_candidates:
                self.values[state] = np.max([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
            
            delta = np.max(np.abs(self.values - old_values))
        
        # Policy Improvement
        for state in range(self.grid_world.get_state_space()):
            self.policy[state] = np.argmax([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
   
    
    def improved_inplace_dynamic_programming(self):
        # Policy Evaluation
        delta = self.threshold + 1
        state_candidates = np.arange(self.grid_world.get_state_space())
        while delta >= self.threshold:
            old_values = self.values.copy()

            
            for state in state_candidates:
                self.values[state] = np.max([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
            
            delta = np.max(np.abs(self.values - old_values))
            state_candidates = np.argsort(np.abs(self.values - old_values))[::-1]
        
        # Policy Improvement
        for state in range(self.grid_world.get_state_space()):
            self.policy[state] = np.argmax([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])

    
    def prioritised_sweeping(self):
        pqueue = PriorityQueue()
        for state in range(self.grid_world.get_state_space()):
            pqueue.put((-np.inf, state))

        next_states = [[] for _ in range(self.grid_world.get_state_space())]
        q_values = [{} for _ in range(self.grid_world.get_state_space())]
        
        for state in range(self.grid_world.get_state_space()):
            for action in range(self.grid_world.get_action_space()):
                next_state, reward, done = self.grid_world.step(state, action)
                next_states[next_state].append(state)

        while True:
            # Get a state
            error, state = pqueue.get()
            old_value = self.values[state]
            error = -error
            if error < self.threshold:
                break

            # V(s) <- max_a(R(s, a) + gamma * V(s'))
            q_value_list = []
            for action in range(self.grid_world.get_action_space()):
                q_value = self.get_q_value(state, action)
                q_value_list.append(q_value)
                q_values[state][action] = q_value
            self.values[state] = np.max(q_value_list)
            
            error = np.abs(old_value - self.values[state])
            for prev_state in next_states[state]:
                pqueue.put((-error, prev_state))

        # Policy Improvement
        for state in range(self.grid_world.get_state_space()):
            self.policy[state] = np.argmax([q_values[state][action] for action in range(self.grid_world.get_action_space())])
    

    def real_time_dynamic_programming(self):
        # Policy Evaluation
        delta = self.threshold + 1
        while (delta >= self.threshold):
            old_values = self.values.copy()

            # Reset the environment
            state = 0
            is_done = False
            visited = [[state, is_done]]

            # Run until the episode is done
            while not is_done:
                qvalue_list, next_state_list, done_list = [], [], []
                for action in range(self.grid_world.get_action_space()):
                    next_state, reward, done = self.grid_world.step(state, action)
                    q_value = reward + self.discount_factor * self.values[next_state] * (1 - done)
                    qvalue_list.append(q_value)
                    next_state_list.append(next_state)
                    done_list.append(done)
                
                self.values[state] = np.max(qvalue_list)
                state = next_state_list[np.argmax(qvalue_list)]
                is_done = done_list[np.argmax(qvalue_list)]
                visited.append([state, is_done])

            # Since the exact start state is unknown, we still use delta here
            visited_states = np.unique([state for state, is_done in visited])
            delta = np.max(np.abs(self.values[visited_states] - old_values[visited_states]))

        # Policy Improvement
        for state in range(self.grid_world.get_state_space()):
            self.policy[state] = np.argmax([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
    

    
    def my_async_dynamic_programming(self):
        # Policy Evaluation and Improvement
        delta = self.threshold + 1
        state_candidates = np.arange(self.grid_world.get_state_space())
        while delta >= self.threshold:
            old_values = self.values.copy()

            for state in state_candidates:
                qvalue_list = [self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())]
                self.values[state] = np.max(qvalue_list)
                self.policy[state] = np.argmax(qvalue_list)
            
            delta = np.max(np.abs(self.values - old_values))
            state_candidates = [state for state in state_candidates if np.abs(self.values[state] - old_values[state]) >= self.threshold]


    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        # raise NotImplementedError

        # 1. In-place Dynamic Programming
        # self.inplace_dynamic_programming()

        # 1-1. Improved In-place Dynamic Programming
        self.improved_inplace_dynamic_programming()

        # 2. Prioritised Sweeping
        # self.prioritised_sweeping()

        # 3. Real-time Dynamic Programming
        # self.real_time_dynamic_programming()

        # 4. My Async Dynamic Programming
        # self.my_async_dynamic_programming()


