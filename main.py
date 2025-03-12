import random
import string
import gzip
from typing import List
import dspy

# Configure DSPy with OpenRouter and timeout
lm = dspy.LM(
    "openrouter/google/gemini-2.0-flash-001", max_tokens=40, timeout=10, cache=False
)
dspy.configure(lm=lm)

# Validate configuration
assert isinstance(lm, dspy.LM), "LM configuration failed"


def create_agent(chromosome: str) -> dict:
    """Create a new agent as a dictionary"""
    # Validate and normalize chromosome
    if isinstance(chromosome, list):
        chromosome = "".join(chromosome)
    chromosome = str(chromosome).strip()[:40]  # Enforce max length

    # Boundary condition assertions
    assert len(chromosome) >= 1, "Agent chromosome cannot be empty"
    assert len(chromosome) <= 40, f"Chromosome length {len(chromosome)} exceeds max"
    assert all(
        c in string.ascii_letters + " " for c in chromosome
    ), "Invalid characters in chromosome"
    if not chromosome:
        # Fallback to random chromosome if empty
        length = random.randint(20, 40)
        chromosome = "".join(random.choices(string.ascii_letters + " ", k=length))
    return {"chromosome": chromosome, "fitness": 0.0}


def evaluate_agent(agent: dict, _problem_description: str) -> float:
    """Evaluate the agent based on the optimization target"""
    # Ensure chromosome is a string
    chromosome = str(agent["chromosome"])

    # Calculate fitness per spec: +1 per 'a' in first 23, -1 per char beyond 23
    fitness = 0.0

    # First 23 characters: +1 for each 'a' (case-insensitive)
    first_part = chromosome[:23].lower()
    a_count = first_part.count("a")
    fitness += a_count

    # After 23: -1 per character
    remaining = chromosome[23:]
    fitness -= len(remaining)

    # Length enforced by truncation in create_agent
    assert (
        len(chromosome) <= 40
    ), f"Chromosome length {len(chromosome)} exceeds maximum allowed"

    # Allow negative fitness as per spec
    agent["fitness"] = fitness
    return agent["fitness"]


def initialize_population(pop_size: int) -> List[dict]:
    """Create initial population with random chromosomes using vectorized operations"""
    # Generate lengths first for vectorization
    lengths = [random.randint(20, 40) for _ in range(pop_size)]
    # Batch create all chromosomes
    chromosomes = [
        "".join(random.choices(string.ascii_letters + " ", k=length))
        for length in lengths
    ]
    # Parallel create agents
    return [create_agent(c) for c in chromosomes]


def select_parents(population: List[dict]) -> List[dict]:
    """Select top 50% of population as parents"""
    sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
    return sorted_pop[: len(population) // 2]


def mutate(chromosome: str) -> str:
    """Mutate a chromosome by replacing one random character"""
    if not chromosome:
        raise ValueError("Cannot mutate empty chromosome")

    # Try up to 5 times to get a valid mutation
    for _ in range(5):
        idx = random.randint(0, len(chromosome) - 1)
        original_char = chromosome[idx]
        # Get a different random character
        # Bias mutation towards adding 'a's
        new_char = random.choice(
            ["a"] * 10
            + [
                c for c in string.ascii_letters + " " if c not in (original_char, "a")
            ]  # Stronger a bias
        )
        new_chromosome = chromosome[:idx] + new_char + chromosome[idx + 1 :]

        if new_chromosome != chromosome:
            break

    # Validate mutation result with more debug info
    assert len(new_chromosome) == len(
        chromosome
    ), f"Length changed from {len(chromosome)} to {len(new_chromosome)}"
    assert (
        new_chromosome != chromosome
    ), f"Mutation failed after 5 attempts: {chromosome}"

    return new_chromosome


def crossover(parent1: dict, parent2: dict) -> dict:
    """Create child by combining parts of parent chromosomes"""
    split = random.randint(1, len(parent1["chromosome"]) - 1)
    new_chromosome = parent1["chromosome"][:split] + parent2["chromosome"][split:]
    return create_agent(new_chromosome)


def run_genetic_algorithm(
    problem: str,
    generations: int = 10,
    pop_size: int = 5,
    mutation_rate: float = 0.5,
    log_file: str = "evolution.log.gz",
):
    """Run genetic algorithm with optimized logging and scaling"""
    assert pop_size > 1, "Population size must be greater than 1"
    assert generations > 0, "Number of generations must be positive"

    population = initialize_population(pop_size)
    assert (
        len(population) == pop_size
    ), f"Population size mismatch {len(population)} != {pop_size}"

    # Clear log file at start
    with gzip.open(log_file, "wt", encoding="utf-8") as f:
        f.write("generation,best_fitness,best_chromosome\n")

    for generation in range(generations):
        # Evaluate all agents
        for agent in population:
            evaluate_agent(agent, problem)

        # Calculate population statistics
        fitness_scores = [agent["fitness"] for agent in population]
        mean_fitness = sum(fitness_scores) / len(fitness_scores)
        sorted_fitness = sorted(fitness_scores)
        median_fitness = sorted_fitness[len(fitness_scores) // 2]
        std_fitness = (
            sum((x - mean_fitness) ** 2 for x in fitness_scores) / len(fitness_scores)
        ) ** 0.5

        # Get best/worst before logging
        sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
        best = sorted_pop[0]
        worst = sorted_pop[-1]

        # Compressed logging
        with gzip.open(log_file, "at", encoding="utf-8") as f:
            f.write(f"{generation},{best['fitness']},{best['chromosome']}\n")

        # Information-dense output
        print(f"\nGen {generation+1}/{generations} | Pop: {pop_size}")
        print(f"Best: {best['chromosome'][:23]}... (fit:{best['fitness']:.1f})")
        print(f"Worst: {worst['chromosome'][:23]}... (fit:{worst['fitness']:.1f})")
        print(
            f"Stats: μ={mean_fitness:.1f} ±{std_fitness:.1f} | Med={median_fitness:.1f}"
        )

        # Validate population state
        assert best["fitness"] >= worst["fitness"], "Fitness ordering invalid"
        assert len(best["chromosome"]) <= 40, "Chromosome exceeded max length"
        assert len(worst["chromosome"]) <= 40, "Chromosome exceeded max length"

        # Select parents and create next generation
        parents = select_parents(population)
        next_gen = parents.copy()

        # Create children through crossover
        while len(next_gen) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            next_gen.append(child)

        # Mutate children based on rate
        for i in range(len(next_gen)):
            if random.random() < mutation_rate:
                next_gen[i]["chromosome"] = mutate(next_gen[i]["chromosome"])

        # Use LLM to improve top candidates (only every 5 generations)
        if generation % 5 == 0:
            for i in range(min(2, len(next_gen))):
                # Use DSPy predictor for guided improvement
                improve_prompt = dspy.Predict(
                    "original_chromosome, problem_description -> improved_chromosome"
                )
                try:
                    response = improve_prompt(
                        original_chromosome=next_gen[i]["chromosome"],
                        problem_description=problem,
                    )
                    # Validate and apply response
                    if (
                        response.completion
                        and len(response.completion) > 0
                        and len(response.completion) <= 40
                        and all(
                            c in string.ascii_letters + " " for c in response.completion
                        )
                    ):
                        next_gen[i]["chromosome"] = response.completion.strip()[:40]
                    else:
                        # If invalid response, mutate instead
                        next_gen[i]["chromosome"] = mutate(next_gen[i]["chromosome"])
                except (TimeoutError, RuntimeError) as e:
                    print(f"LLM improvement failed: {str(e)}")
                    # If LLM fails, mutate instead
                    next_gen[i]["chromosome"] = mutate(next_gen[i]["chromosome"])

        population = next_gen


if __name__ == "__main__":
    PROBLEM = (
        "Generate a string with MAXIMUM 'a's in first 23 characters, "
        "then keep it short. STRICTLY prioritize 'a's over all other considerations!"
    )
    run_genetic_algorithm(PROBLEM, generations=20, pop_size=10)
