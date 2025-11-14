import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# -------------------- Problem Definitions --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str  # 'bit' or 'real'
    dim: int
    bounds: Optional[Tuple[float, float]]
    fitness_fn: Callable[[np.ndarray], float]


def make_onemax(dim: int) -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        return float(np.sum(x))  # maximize number of ones

    return GAProblem(
        name=f"OneMax ({dim} bits)",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )


def make_custom_80_50() -> GAProblem:
    """
    Custom 80-bit problem where fitness reaches maximum 80 when number of ones == 50.
    Fitness formula: f(x) = 80 - abs(50 - num_ones)
    This yields a maximum value of 80 at num_ones == 50 and decreases linearly as the Hamming weight
    deviates from 50.
    """

    def fitness(x: np.ndarray) -> float:
        num_ones = int(np.sum(x))
        return 80.0 - abs(50 - num_ones)

    return GAProblem(
        name="Custom 80-bit (peak at 50 ones)",
        chromosome_type="bit",
        dim=80,
        bounds=None,
        fitness_fn=fitness,
    )


# -------------------- GA Operators --------------------
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    if problem.chromosome_type == "bit":
        return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)
    else:
        assert problem.bounds is not None
        lo, hi = problem.bounds
        return rng.uniform(lo, hi, size=(pop_size, problem.dim))


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return int(best)


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)


def run_ga(
        problem: GAProblem,
        pop_size: int,
        generations: int,
        crossover_rate: float,
        mutation_rate: float,
        tournament_k: int,
        elitism: int,
        seed: Optional[int] = None,
        stream_live: bool = True,
):
    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    # Live UI containers
    chart_area = st.empty()
    best_area = st.empty()

    history_best: List[float] = []
    history_avg: List[float] = []
    history_worst: List[float] = []

    for gen in range(generations):
        # Logging
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))
        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        # Live updates
        if stream_live:
            df = pd.DataFrame(
                {
                    "Best": history_best,
                    "Average": history_avg,
                    "Worst": history_worst,
                }
            )
            chart_area.line_chart(df)
            best_area.markdown(
                f"Generation {gen + 1}/{generations} — Best fitness: **{best_fit:.6f}**"
            )

        # Elitism: keep top E
        E = max(0, min(elitism, pop_size))
        elite_idx = np.argpartition(fit, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy() if E > 0 else np.empty((0, pop.shape[1]))

        # Create next generation
        next_pop: List[np.ndarray] = []
        while len(next_pop) < pop_size - E:
            # Select parents
            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            # Crossover
            if rng.random() < crossover_rate:
                if problem.chromosome_type == "bit":
                    c1, c2 = one_point_crossover(p1, p2, rng)
                else:
                    # fallback: copy parents for real-valued
                    c1, c2 = p1.copy(), p2.copy()
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            if problem.chromosome_type == "bit":
                c1 = bit_mutation(c1, mutation_rate, rng)
                c2 = bit_mutation(c2, mutation_rate, rng)
            else:
                c1 = c1
                c2 = c2

            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        # Insert elites and finalize
        pop = np.vstack([np.array(next_pop), elites]) if E > 0 else np.array(next_pop)
        fit = evaluate(pop, problem)

    # Final metrics and best solution
    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])
    df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})

    return {
        "best": best,
        "best_fitness": best_fit,
        "history": df,
        "final_population": pop,
        "final_fitness": fit,
    }


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm - Custom 80-bit", page_icon="🦖", layout="wide")
st.title("Genetic Algorithm (GA) — Custom 80-bit target")
st.caption("Evolves 80-bit individuals; peak fitness 80 when exactly 50 ones are present.")

with st.sidebar:
    st.header("Problem")

    problem_type = st.selectbox(
        "Type",
        ["Custom 80-bit (peak 50)", "OneMax (bits)"], index=0)

    if problem_type == "OneMax (bits)":
        dim = st.number_input("Chromosome length (bits)", min_value=8, max_value=4096, value=64, step=8)
        problem = make_onemax(int(dim))
    else:
        # Use the required custom problem defaults
        problem = make_custom_80_50()

    st.header("GA Parameters")
    pop_size = st.number_input("Population size", min_value=10, max_value=5000, value=300, step=10)
    generations = st.number_input("Generations", min_value=1, max_value=10000, value=50, step=1)
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.9, 0.05)
    mutation_rate = st.slider("Mutation rate (per gene)", 0.0, 1.0, 0.01, 0.005)
    tournament_k = st.slider("Tournament size", 2, 10, 3)
    elitism = st.slider("Elites per generation", 0, 100, 2)
    seed = st.number_input("Random seed (optional)", min_value=0, max_value=2 ** 32 - 1, value=42)
    live = st.checkbox("Live chart while running", value=True)

left, right = st.columns([1, 1])

with left:
    if st.button("Run GA", type="primary"):
        result = run_ga(
            problem=problem,
            pop_size=int(pop_size),
            generations=int(generations),
            crossover_rate=float(crossover_rate),
            mutation_rate=float(mutation_rate),
            tournament_k=int(tournament_k),
            elitism=int(elitism),
            seed=int(seed),
            stream_live=bool(live),
        )

        # Store final population for the snapshot view
        st.session_state["_final_pop"] = result["final_population"]
        st.session_state["_final_fit"] = result["final_fitness"]

        st.subheader("Fitness Over Generations")
        st.line_chart(result["history"])

        st.subheader("Best Solution")
        st.write(f"Best fitness: {result['best_fitness']:.6f}")

        if problem.chromosome_type == "bit":
            bitstring = ''.join(map(str, result['best'].astype(int).tolist()))
            st.code(bitstring, language="text")
            st.write(f"Number of ones: {int(np.sum(result['best']))} / {problem.dim}")
        else:
            vec = result['best'].astype(float)
            st.write("x* =", np.array2string(vec, precision=4, suppress_small=True))

with right:
    st.subheader("Population Snapshot (final)")
    st.caption("Shows first 20 individuals with fitness")
    if st.button("Show final population table"):
        pop = st.session_state.get("_final_pop")
        fit = st.session_state.get("_final_fit")
        if pop is None or fit is None:
            st.info("Run GA first to view the final population.")
        else:
            nshow = min(20, pop.shape[0])
            df = pd.DataFrame(pop[:nshow])
            df["fitness"] = fit[:nshow]
            st.dataframe(df, use_container_width=True)

# Initialize session state storage keys if absent
if "_final_pop" not in st.session_state:
    st.session_state["_final_pop"] = None
    st.session_state["_final_fit"] = None