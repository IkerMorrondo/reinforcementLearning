########
### SIMPLE VALUE ITERATION IMPLEMENTATION
########

import matplotlib.pyplot as plt
import numpy as np
from env import GridworldEnv

def value_iteration_step(V, P, gamma):
    """Performs a single Value Iteration pass using Bellman Optimality Equation (BOE)
    
    Inputs:
      V: current state-value function estimate from iteration k
      P: Environment's transition model (gymnasium's env.P format)
      gamma: discount factor
    Returns:
      V_new: updated state-value function for iteration k+1
      Q_new: computed state-action value for iteration k+1
      delta: max value change across all states regarding the state-values
    """

    n_states = len(P)
    n_actions = len(P[0])

    # TODO: initialize V_new and Q_new to zeros (np.zeros) 
    V_new = np.zeros(n_states)
    Q_new = np.zeros((n_states, n_actions))
    delta = 0

    # TODO: apply BOE to obtain updated V_new (and Q_new) for all states and actions

    # TODO: obtain the max difference for all possible states V_new(s) - V(s) (not the mean)

    for s in range(n_states):
        for a in range(n_actions):
            Q_new[s, a] = sum(
                prob * (reward + gamma * V[next_state])
                for prob, next_state, reward, _ in P[s][a]
            )
        V_new[s] = np.max(Q_new[s])
        delta = max(delta, abs(V_new[s] - V[s]))


    

    return V_new, Q_new, delta

def extract_policy(Q, P):
    """
    policy extraction function

    Inputs: 
      Q: desired state-action function
      P: Environment's transition model (gymnasium's env.P format)
    Returns:
      policy: array mapping states to actions
    """
    

    n_states = len(P)

    # TODO: initialize policy to zeros (in this case int type, not float values)
    policy = np.zeros(n_states, dtype=int)


    # TODO: iterate over all states and select action that maximizes Q(s) (np.argmax)
    for s in range(n_states):
        policy[s] = np.argmax(Q[s])


    return policy

def generate_graph(value_history):
    """
    Auxiliary function to generate graph of state-values for each iteration. No need to check.
    Only works on basic env
    """
    name_list = (
        "(1,3)",
        "(2,3)",
        "(3,3)",
        "+1 Goal",
        "(1,2)",
        "Wall (#)",
        "(3,2)",
        "-1 Pit",
        "(1,1)",
        "(2,1)",
        "(3,1)",
        "(4,1)",
    )

    color_list = (
        "tab:blue",
        "tab:orange",
        "tab:green",
        "black",  # Goal line
        "tab:purple",
        "gray",  # Wall line
        "tab:cyan",
        "tab:red",  # Pit line
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "mediumpurple",
    )

    iterations = np.arange(len(value_history))

    plt.figure(figsize=(10, 6))

    for state in range(12):
        state_values = [v[state] for v in value_history]
        # Use dashed lines for terminal/wall states so non-terminal curves stay clear
        linestyle = (
            "--" if state in (3, 5, 7) else "-"
        )  # Goal (3), Wall (5), Pit (7)
        plt.plot(
            iterations,
            state_values,
            color=color_list[state],
            label=name_list[state],
            linestyle=linestyle,
            linewidth=2 if state not in (3, 5, 7) else 1,
        )

    # Place legend OUTSIDE the axes to avoid hiding data
    plt.legend(
        title="Coordinates (x, y)\n(Left → Right, Bottom → Up)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )

    # plt.ylim((-1.1, +1.1))
    plt.xlim((0, len(value_history)-1))
    plt.ylabel("Value $V(s)$", fontsize=13)
    plt.xlabel("Iteration", fontsize=13)
    plt.title("Value Iteration State Convergence", fontsize=14, pad=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    # plt.savefig("./output.jpg", dpi=300)
    plt.show()

def value_iteration(env, gamma=0.999, epsilon=1e-8):
    """
    Whole value iteration algorithm. Uses the previous value_iteration_step and extract_policy functions
    Inputs:
      env: whole gymnasium or basic env to extract multiple required parameters
      gamma: discount factor
      epsilon: desired number of declaring convergence
    Ouputs:
      V_new: last optimal value-function
      Q: last optimal Q-function
      policy: optimal policy extracted by Q-function
      value_history: list of all appended V variables (appended at each iteration)
    """

    n_states = len(env.P)
    value_history = []  # just to plot how value-function converges

    iteration = 0
    V_new = np.zeros(n_states)

    # TODO: main loop of algorithm, iterate infinitely and use value_iteration_step until convergence.
    while True:
        V_new, Q, delta = value_iteration_step(V_new, env.P, gamma)
        value_history.append(V_new.copy())
        iteration += 1

        if delta < epsilon:
            break
    policy = extract_policy(Q, env.P)

    return V_new, Q, policy, value_history

def main():
    import tools
    import gymnasium as gym

    ### VALUE ITERATION IN BASIC ENV

    ### select one of the 2:
    env = GridworldEnv(is_slippery=False)
    # env = GridworldEnv(is_slippery=True)

    V, Q, policy, value_history = value_iteration(env)

    # NOTE: check resulting value function and policy to evaluate correctness
    tools.plot_value_function(V, env)
    tools.plot_policy(policy, env)

    # NOTE: generate graph of convergence for each value-state throughout algorithm iterations
    generate_graph(value_history)


    # TODO: implement the same but for gymnasium's frozenlake env
    # main differences:
    # - use env.unwrapped.P instead of env.P
    # - can't call generate_graph (function only implemented for basic env)

    # # VALUE ITERATION IN FROZENLAKE ENV
    # # select one of the 2:
    # env = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8")
    # # env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8")
    env = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8")
    
    env.P = env.unwrapped.P 

    V_fl, Q_fl, policy_fl, value_history_fl = value_iteration(env)

    tools.plot_value_function(V_fl, env)
    tools.plot_policy(policy_fl, env)

if __name__ == "__main__":
    main()
