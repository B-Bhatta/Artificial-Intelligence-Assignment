"""Hidden Markov Model - Viterbi Algorithm

This program predicts the most likely hidden state sequence for a
small Hidden Markov Model using the Viterbi algorithm.

Hidden states:
    Sunny, Rainy

Observable events:
    Walk, Shop, Clean

Run:
    python hmm_viterbi.py
"""

import matplotlib
matplotlib.use("Agg")  # Allows plot generation without opening a GUI window
import matplotlib.pyplot as plt


states = ["Sunny", "Rainy"]
observations = ["Walk", "Walk", "Shop", "Clean"]

initial_probability = {
    "Sunny": 0.6,
    "Rainy": 0.4,
}

transition_probability = {
    "Sunny": {"Sunny": 0.7, "Rainy": 0.3},
    "Rainy": {"Sunny": 0.4, "Rainy": 0.6},
}

emission_probability = {
    "Sunny": {"Walk": 0.6, "Shop": 0.3, "Clean": 0.1},
    "Rainy": {"Walk": 0.1, "Shop": 0.2, "Clean": 0.7},
}


def viterbi(obs_sequence, hidden_states, start_prob, transition_prob, emission_prob):
    """Return the most likely state path and probability table."""
    viterbi_table = [{}]
    backpointer = [{}]

    # Initialization step
    first_observation = obs_sequence[0]
    for state in hidden_states:
        viterbi_table[0][state] = start_prob[state] * emission_prob[state][first_observation]
        backpointer[0][state] = None

    # Recursion step
    for time_index in range(1, len(obs_sequence)):
        current_observation = obs_sequence[time_index]
        viterbi_table.append({})
        backpointer.append({})

        for current_state in hidden_states:
            best_previous_state = None
            best_probability = -1

            for previous_state in hidden_states:
                probability = (
                    viterbi_table[time_index - 1][previous_state]
                    * transition_prob[previous_state][current_state]
                    * emission_prob[current_state][current_observation]
                )

                if probability > best_probability:
                    best_probability = probability
                    best_previous_state = previous_state

            viterbi_table[time_index][current_state] = best_probability
            backpointer[time_index][current_state] = best_previous_state

    # Termination step
    final_time_index = len(obs_sequence) - 1
    final_state = max(viterbi_table[final_time_index], key=viterbi_table[final_time_index].get)

    # Backtracking step
    best_path = [final_state]
    for time_index in range(final_time_index, 0, -1):
        best_path.insert(0, backpointer[time_index][best_path[0]])

    return best_path, viterbi_table


def generate_probability_plot(viterbi_table, obs_sequence, output_file="hmm_probability_plot.png"):
    """Generate a probability plot for each hidden state over time."""
    time_steps = list(range(len(obs_sequence)))

    plt.figure(figsize=(8, 5))
    for state in states:
        state_probabilities = [viterbi_table[t][state] for t in time_steps]
        plt.plot(time_steps, state_probabilities, marker="o", label=state)

    plt.title("Viterbi Probability Plot")
    plt.xlabel("Time Step / Observation")
    plt.ylabel("Viterbi Probability")
    plt.xticks(time_steps, [f"T={i}\n{obs}" for i, obs in enumerate(obs_sequence)])
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


if __name__ == "__main__":
    predicted_path, probability_table = viterbi(
        observations,
        states,
        initial_probability,
        transition_probability,
        emission_probability,
    )

    print("Hidden Markov Model - Viterbi Algorithm Execution")
    print()
    print("-" * 50)
    print(f"{'Time Step':<16}| {'Observation':<16}| Predicted State")
    print("-" * 50)

    for index, (observation, state) in enumerate(zip(observations, predicted_path)):
        print(f"{f'T = {index}':<16}| {observation:<16}| {state}")

    print("-" * 50)

    generate_probability_plot(probability_table, observations)
    print("Graphical visualization saved as: hmm_probability_plot.png")
