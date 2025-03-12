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

    # First 23 characters: +2 for 'a's, -2 for other letters (case-insensitive)
    first_part = chromosome[:23].lower()
    a_count = first_part.count("a")
    other_count = len(first_part) - a_count
    fitness += (a_count * 2) - (other_count * 2)
    
    # Debug assertions
    assert a_count >= 0, f"Invalid a_count {a_count} for {chromosome}"
    assert other_count >= 0, f"Invalid other_count {other_count} for {chromosome}"
    if a_count == 0:
        print(f"WARNING: No 'a's in first 23 of: {chromosome}")

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
    """Select parents using Pareto distribution weighted by fitness"""
    # Calculate Pareto weights (power law distribution)
    fitness_values = [agent["fitness"] for agent in population]
    max_fitness = max(fitness_values)
    min_fitness = min(fitness_values)
    
    # Normalize and apply quadratic weighting
    spread = max_fitness - min_fitness if max_fitness != min_fitness else 1
    weights = [((f - min_fitness)/spread)**2 for f in fitness_values]
    
    # Handle zero weights case
    if sum(weights) <= 0:
        return random.sample(population, k=len(population)//2)
        
    # Weighted selection without replacement using reservoir sampling
    selected = []
    for _ in range(len(population)//2):
        candidates = random.choices(population, weights=weights, k=5)
        selected.append(max(candidates, key=lambda x: x["fitness"]))
        # Remove selected candidate's weight to prevent duplicates
        idx = population.index(selected[-1])
        weights[idx] = 0
    return list({agent["chromosome"]: agent for agent in selected}.values())  # Deduplicate


def mutate_with_llm(chromosome: str, problem: str) -> str:
    """Mutate chromosome using LLM-based rephrasing"""
    mutate_prompt = dspy.Predict("original_chromosome, problem_description -> mutated_chromosome")
    try:
        response = mutate_prompt(
            original_chromosome=chromosome,
            problem_description=f"{problem} MUTATION RULES:\n1. MAXIMIZE 'a's IN FIRST 23 CHARACTERS\n2. REMOVE ALL CHARACTERS AFTER FIRST 23\n3. USE ONLY LETTERS\n4. MAX LENGTH 40"
        )
        if response.completions:
            mutated = response.completions[0].strip()[:23]  # Strictly keep first 23 chars
            mutated = mutated.ljust(23, 'a')[:40]  # Ensure min length 23, max 40
            mutated = ''.join([c if c.isalpha() else 'a' for c in mutated])  # Enforce letters only
            return mutated
    except (TimeoutError, RuntimeError):
        pass
    return mutate(chromosome)  # Fallback to traditional mutation

def mutate(chromosome: str) -> str:
    """Mutate a chromosome with 30% chance of LLM-based mutation"""
    if random.random() < 0.3:  # 30% chance for LLM mutation
        return mutate_with_llm(chromosome, PROBLEM)
    if not chromosome:
        raise ValueError("Cannot mutate empty chromosome")

    # Try up to 5 times to get a valid mutation
    for _ in range(5):
        idx = random.randint(0, len(chromosome) - 1)
        original_char = chromosome[idx]
        # Get a different random character
        # Bias mutation towards adding 'a's
        new_char = random.choice(
            ["a"] * 50  # Extreme a bias
            + [c for c in string.ascii_letters + " " if c not in (original_char, "a")]
            + ["a"] * 10  # Massive chance to add a's
        )
        assert new_char != original_char, f"Failed mutation at index {idx} of {chromosome}"
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

        # Calculate sliding window statistics (last 100 evaluations)
        window_size = 100
        all_fitness = [agent["fitness"] for agent in population]
        
        # Maintain sliding window in memory
        if not hasattr(run_genetic_algorithm, "fitness_window"):
            run_genetic_algorithm.fitness_window = []
        run_genetic_algorithm.fitness_window.extend(all_fitness)
        run_genetic_algorithm.fitness_window = run_genetic_algorithm.fitness_window[-window_size:]
        
        # Calculate stats on window
        window_fitness = run_genetic_algorithm.fitness_window
        mean_fitness = sum(window_fitness) / len(window_fitness) if window_fitness else 0
        sorted_fitness = sorted(window_fitness)
        median_fitness = sorted_fitness[len(sorted_fitness)//2] if sorted_fitness else 0
        std_fitness = (sum((x-mean_fitness)**2 for x in window_fitness)/len(window_fitness))**0.5 if window_fitness else 0

        # Get best/worst before logging
        sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
        best = sorted_pop[0]
        worst = sorted_pop[-1]

        # Compressed logging
        with gzip.open(log_file, "at", encoding="utf-8") as f:
            f.write(f"{generation},{best['fitness']},{best['chromosome']}\n")

        # Information-dense output
        print(f"\n—— Generation {generation+1}/{generations} ——")
        print(f"Population: {pop_size} agents")
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

        # Create children through crossover with parent validation
        while len(next_gen) < pop_size:
            if len(parents) >= 2:
                parent1, parent2 = random.sample(parents, 2)
                child = crossover(parent1, parent2)
            else:
                # Handle insufficient parents by mutating existing members
                parent = random.choice(parents) if parents else create_agent("")
                child = create_agent(mutate(parent["chromosome"]))
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
                        problem_description=f"STRICTLY use ONLY 'a's in first 23 characters! REPLACE ALL other characters with 'a's! {problem}",
                    )
                    # Validate and apply response
                    if (
                        response.completions[0]
                        and len(response.completions[0]) > 0
                        and len(response.completions[0]) <= 40
                        and all(
                            c in string.ascii_letters + " " for c in response.completions[0]
                        )
                    ):
                        next_gen[i]["chromosome"] = response.completions[0].strip()[:40]
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
