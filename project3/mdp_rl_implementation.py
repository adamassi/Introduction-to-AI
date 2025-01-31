from copy import deepcopy
from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np

def get_states(mdp):
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if mdp.board[row][col] != 'WALL':
                yield (row, col)

def get_state_reward(mdp, state):
    row, col = state
    if mdp.board[row][col] == 'WALL':
        return None
    return float(mdp.board[row][col])

def get_initialized_utility(mdp):
    u_init = [[0.0] * mdp.num_col for _ in range(mdp.num_row)]
    for terminal_state in mdp.terminal_states:
        row, col = terminal_state
        u_init[row][col] = get_state_reward(mdp, terminal_state)  # terminal state utility is its reward value only

    return u_init


def get_state_util_under_policy(mdp, policy, util, state):
    row, col = state
    neigh_states = [mdp.step(state, action) for action in mdp.actions]

    # Convert the policy action from string to Action enum
    policy_action_str = policy[row][col]
    policy_action = Action[policy_action_str] if isinstance(policy_action_str, str) else policy_action_str

    neigh_probs = mdp.transition_function[policy_action]
    state_reward = get_state_reward(mdp, state)

    # R(s) + gamma * sum(P(s'|s,policy(s)) * U(s') for s' in successors)
    avg_action_util = sum(prob * util[neigh[0]][neigh[1]] for prob, neigh in zip(neigh_probs, neigh_states))
    return state_reward + mdp.gamma * avg_action_util, avg_action_util


def bellman_eq(mdp, util, state, epsilon=10 ** (-3)):
    state_reward = get_state_reward(mdp, state)
    max_avg_util = float("-inf")
    best_action = None
    next_states = [mdp.step(state, action) for action in mdp.actions]
    avg_utils = []

    if state in mdp.terminal_states:
        return state_reward, None, None, []

    for action in mdp.actions:
        avg_util_from_action = sum(prob * util[next_state[0]][next_state[1]]
                                   for next_state, prob in zip(next_states, mdp.transition_function[action]))
        if avg_util_from_action > max_avg_util:
            best_action = action
            max_avg_util = avg_util_from_action
        avg_utils.append(avg_util_from_action)

    x = len(str(epsilon).split(".")[1]) + 1

    all_best_actions = [action for action, avg_util in zip(mdp.actions, avg_utils)
                        if abs(round(max_avg_util, x) - round(avg_util, x)) < epsilon]

    return state_reward + mdp.gamma * max_avg_util, best_action, max_avg_util, all_best_actions

def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float = 10 ** (-3)) -> np.ndarray:
    next_util = deepcopy(U_init)

    while True:
        U_final = deepcopy(next_util)
        delta = 0
        states = get_states(mdp)
        for state in states:
            row, col = state
            next_util[row][col], _, _, _ = bellman_eq(mdp, U_final, state)
            delta = max(delta, abs(next_util[row][col] - U_final[row][col]))
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break

    return U_final

def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    policy = [[None] * mdp.num_col for _ in range(mdp.num_row)]

    states = get_states(mdp)
    for state in states:
        if state in mdp.terminal_states:
            continue  # policy must give terminal states None as the action
        row, col = state
        _, best_action, _, _ = bellman_eq(mdp, U, state)
        policy[row][col] = best_action

    return policy

def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:
    u_init = get_initialized_utility(mdp)

    epsilon = 10 ** (-3)  # error tolerance, like in value iteration
    util = deepcopy(u_init)

    while True:
        next_util = deepcopy(u_init)
        delta = 0
        states = get_states(mdp)
        for state in states:
            if state in mdp.terminal_states:
                continue  # utility was already initialized as state reward
            row, col = state
            next_util[row][col], _ = get_state_util_under_policy(mdp, policy, util, state)
            delta = max(delta, abs(next_util[row][col] - util[row][col]))
        util = next_util
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break

    return util

def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:
    optimal_policy = policy_init
    changed = True

    while changed:
        util = policy_evaluation(mdp, optimal_policy)
        changed = False
        states = get_states(mdp)
        for state in states:
            row, col = state
            if state in mdp.terminal_states:
                optimal_policy[row][col] = None
                continue
            _, best_action, max_action_util, _ = bellman_eq(mdp, util, state)
            _, curr_action_util = get_state_util_under_policy(mdp, optimal_policy, util, state)
            if max_action_util > curr_action_util:
                optimal_policy[row][col] = best_action
                changed = True

    return optimal_policy

#
def adp_algorithm(
    sim: Simulator,
    num_episodes: int,
    num_rows: int = 3,
    num_cols: int = 4,
    actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
) -> Tuple[np.ndarray, Dict[str, Dict[Action, float]]]:
    """
    Runs the ADP algorithm given the simulator, the number of rows and columns in the grid,
    the list of actions, and the number of episodes.
    """

    # Initialize the reward matrix with zeros
    reward_matrix = np.zeros((num_rows, num_cols))

    # Initialize the transition probabilities dictionary
    transition_probs = {a: {b: 0.0 for b in actions} for a in actions}

    # For each episode
    for episode_gen in sim.replay(num_episodes=num_episodes):
        for step in episode_gen:
            state, reward, action, actual_action = step

            if state is None:
                continue

            # Update the reward matrix
            r, c = state  # Assuming state is a tuple (row, col)
            reward_matrix[r, c] = reward

            # Check if action and actual_action are not None before updating transition probabilities
            if action is not None and actual_action is not None:
                if action in transition_probs and actual_action in transition_probs[action]:
                    transition_probs[action][actual_action] += 1.0  # Increment the count

    # Normalize the transition probabilities
    for action in actions:
        total = sum(transition_probs[action].values())
        if total > 0:
            for actual_action in actions:
                transition_probs[action][actual_action] /= total  # Convert counts to probabilities

     # Replace 0s with None in the reward matrix
    reward_matrix = np.where(reward_matrix == 0, None, reward_matrix)

    # Convert to the desired format
    # transition_probs = {str(outer_action): {actual_action: prob for actual_action, prob in inner_dict.items()}
    #                     for outer_action, inner_dict in t_probs.items()}

    return reward_matrix, transition_probs



# if __name__ == '__main__':
#     sim = Simulator()
#     reward_matrix, transition_probabilities = adp_algorithm(sim, num_episodes=1)
#     print("Reward Matrix:")
#     print(reward_matrix)
#     print("Transition Probabilities:")
#     print(transition_probabilities)