########
### SIMPLE POLICY ITERATION IMPLEMENTATION
########

import matplotlib.pyplot as plt
import numpy as np
from env import GridworldEnv

def evaluate_policy(V, policy, P, gamma, epsilon):
    """Inner Loop (j): Iterative Policy Evaluation.

    Computes the exact value function V^{pi_k} for the fixed policy pi_k
    until Delta <= epsilon.
    """
    n_states = len(P)
    V = V.copy()

    # TODO: inner loop until convergence of value-function
    #  apply Bellman equation (not BOE) at each iteration


    return V


def improve_policy(V, P, gamma):
    """Outer Step (k): Policy Improvement.

    Computes Q_pi_k(s, a) using the evaluated V_pi_k and extracts the new greedy
    action argmax_a Q(s, a) for every state.
    """
    n_states = len(P)
    n_actions = len(P[0])

    Q = np.zeros((n_states, n_actions))
    new_policy = np.zeros(n_states, dtype=int)

    # TODO: loop over all states and actions obtaining Q(s,a) and apply argmax to decide best actions

    return new_policy, Q


def generate_graph(value_history):
    """Generate convergence graph of state values across outer Policy Iteration steps."""
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
        linestyle = "--" if state in (3, 5, 7) else "-"
        plt.plot(
            iterations,
            state_values,
            color=color_list[state],
            label=name_list[state],
            linestyle=linestyle,
            linewidth=2 if state not in (3, 5, 7) else 1,
            marker="o",  # Markers help highlight the few outer steps
        )

    plt.legend(
        title="Coordinates (x, y)\n(Left → Right, Bottom → Up)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )

    plt.xlim((0, len(value_history) - 1))
    plt.ylabel("True Policy Return $V^{\\pi_k}(s)$", fontsize=13)
    plt.xlabel("Outer Policy Iteration ($k$)", fontsize=13)
    plt.title("Policy Iteration State Convergence", fontsize=14, pad=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    # plt.savefig("./pi_output.jpg", dpi=300)
    plt.show()


def policy_iteration(env, gamma=0.999, epsilon=1e-8):
    """
    Whole policy iteration algorithm. Uses the previous evaluate_policy and improve_policy functions
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
    # TIP: for comparing the previous and next policies, you can use np.array_equal

    n_states = len(env.P)
    value_history = []

    # Initialize V(s) and policy pi(s) arbitrarily
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    iteration = 0
    # TODO: main loop here:

    return V, Q, policy, value_history


def main():
    import tools
    import gymnasium as gym

    # # POLICY ITERATION IN BASIC ENV
    gamma = 0.99
    epsilon = 1e-8

    # # select one of the 2:
    env = GridworldEnv(is_slippery=False)
    # env = GridworldEnv(is_slippery=True)

    V, Q, policy, value_history = policy_iteration(env, gamma=gamma, epsilon=epsilon)

    # NOTE: check resulting value function and policy to evaluate correctness
    tools.plot_value_function(V, env)
    tools.plot_policy(policy, env)
    # NOTE: generate graph of convergence for each value-state throughout algorithm iterations
    generate_graph(value_history)


    # TODO: implement the same but for gymnasium's frozenlake env
    # main differences:
    # - use env.unwrapped.P instead of env.P
    # - can't call generate_graph (function only implemented for basic env)

    # ### select one of the 2:
    # env = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8")
    # # env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8")

if __name__ == "__main__":
    main()
