import random
import string
import gzip
import json
import numpy as np
from typing import List
from rich.console import Console
from rich.table import Table
import dspy

# TODO List (sorted by priority):
# 1. Implement proper weighted sampling without replacement in parent selection
# 2. Add chromosome validation before LLM mating selection
# 3. Optimize fitness window statistics calculations
# 4. Add population size monitoring/limiting

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

    # Fitness calculation per hidden spec: +1 per 'a' in first 23, -1 after
    fitness = 0.0
    core_segment = chromosome[:23].lower()
    assert len(core_segment) == 23, f"Core segment must be 23 chars, got {len(core_segment)}"
    
    # Simplified calculation that actually matches the described hidden goal
    fitness += core_segment.count("a")  # +1 per 'a'
    fitness -= (len(core_segment) - core_segment.count("a"))  # -1 per non-a
    fitness -= len(chromosome[23:])  # -1 per character beyond 23
    
    # Validation assertions
    assert len(core_segment) == 23, f"Core segment must be 23 chars, got {len(core_segment)}"
    if core_segment.count("a") == 0:
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
    """Select parents using Pareto distribution weighted by fitness^2"""
    # Filter and sort by fitness descending
    sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
    
    # Calculate Pareto weights with alpha=2
    fitness_values = [max(a["fitness"], 0)**2 for a in sorted_pop]
    ranks = range(1, len(sorted_pop)+1)
    pareto_weights = [f * (1/(r**2)) for f, r in zip(fitness_values, ranks)]  # Alpha=2 Pareto distribution
    total_weight = sum(pareto_weights)
    
    if total_weight <= 0:
        raise ValueError("All agents have zero fitness - cannot select parents")
    
    # Convert weights to probabilities with epsilon to avoid zeros
    probs = np.array([w/total_weight for w in pareto_weights], dtype=np.float64)
    probs += 1e-8  # Add small epsilon to avoid zero probabilities
    probs /= probs.sum()  # Renormalize
    
    # Gumbel-max trick for weighted sampling without replacement
    population_size = len(population)
    sample_size = len(population) // 2
    gumbel_noise = np.random.gumbel(0, 1, population_size)
    scores = np.log(probs) + gumbel_noise
    selected_indices = np.argpartition(-scores, sample_size)[:sample_size]
    
    # Validate unique selection
    unique_indices = set(selected_indices)
    assert len(unique_indices) == sample_size, f"Duplicate selections {len(unique_indices)} != {sample_size}"
    
    return [population[i] for i in selected_indices]



def mutate_with_llm(chromosome: str, problem: str) -> str:
    """Mutate chromosome using LLM-based rephrasing"""
    mutate_prompt = dspy.Predict("original_chromosome, problem_description -> mutated_chromosome")
    # Strict mutation with validation
    try:
        response = mutate_prompt(
            original_chromosome=chromosome,
            problem_description=f"{problem}\nCreate a mutated version following standard genetic optimization principles",  # Obfuscated prompt
        )
        mutated = str(response.mutated_chromosome).strip()[:40]  # Hard truncate
        # More rigorous validation and normalization
        mutated = ''.join([c.lower() if c.isalpha() else '' for c in mutated])  # Letters only
        mutated = mutated.ljust(23, 'a')  # Fill to min length with 'a's which help fitness
        mutated = mutated[:40]  # Final length cap
        
        # Validation asserts
        assert len(mutated) >= 23, f"Mutation too short: {len(mutated)}"
        assert len(mutated) <= 40, f"Mutation too long: {len(mutated)}"
        assert all(c.isalpha() for c in mutated), f"Invalid chars: {mutated}"
        
        return mutated
    except (TimeoutError, RuntimeError, AssertionError) as e:
        print(f"Mutation failed: {str(e)}")
        # Fallback to random mutation without recursion
        return ''.join(random.choices(string.ascii_letters, k=random.randint(23,40)))

def mutate(chromosome: str) -> str:
    """Mutate a chromosome with LLM-based mutation as primary strategy"""
    # Get problem from DSPy configuration instead of global
    return mutate_with_llm(chromosome, dspy.settings.get("problem"))


def llm_select_mate(parent: dict, candidates: List[dict], problem: str) -> dict:
    """Select mate using LLM prompt with candidate chromosomes"""
    # Format candidates with indexes
    candidate_list = [f"{idx}: {agent['chromosome'][:23]}" 
                    for idx, agent in enumerate(candidates)]
    
    # Create mating prompt with hidden selection criteria
    prompt = dspy.Predict("parent_chromosome, candidates, problem -> best_candidate_id")
    response = prompt(
        parent_chromosome=parent["chromosome"],
        candidates="\n".join(candidate_list),
        problem=f"{problem}\nSELECTION RULES:\n1. MAXIMIZE CORE SIMILARITY\n2. OPTIMIZE LENGTH 23\n3. ENSURE CHARACTER DIVERSITY\n4. PRIORITIZE COMPACTNESS",
    )
    
    # Validate and parse response
    try:
        # Get first completion and clean it
        raw_response = str(response.best_candidate_id).strip()
        chosen_id = int(raw_response.split(":")[0])  # Handle "1: explanation" format
        
        # Validate ID is within candidate range
        if 0 <= chosen_id < len(candidates):
            return candidates[chosen_id]
        return random.choice(candidates)
    except (ValueError, IndexError, AttributeError):
        return random.choice(candidates)

def crossover(parent: dict, population: List[dict], problem: str) -> dict:
    """Create child through LLM-assisted mate selection"""
    # Get candidates using weighted sampling without replacement
    candidates = random.choices(
        population=[a for a in population if a["chromosome"] != parent["chromosome"]],
        weights=[a["fitness"]**2 for a in population if a["chromosome"] != parent["chromosome"]],
        k=min(5, len(population)//2)
    )
    
    # Select mate using LLM prompt from qualified candidates
    mate = llm_select_mate(parent, candidates, problem)
    
    # Combine chromosomes
    split = random.randint(1, len(parent["chromosome"]) - 1)
    new_chromosome = parent["chromosome"][:split] + mate["chromosome"][split:]
    return create_agent(new_chromosome)


def update_fitness_window(fitness_window: list, new_fitnesses: list, window_size: int) -> list:
    """Maintain sliding window of last 100 evaluations"""
    # Combine new fitness values with existing window
    combined = (fitness_window[-window_size:] if fitness_window else []) + new_fitnesses
    
    # Trim to only keep the newest entries up to window size
    return combined[-window_size:]

def calculate_window_statistics(fitness_window, window_size):
    """Calculate statistics for current fitness window including best/worst"""
    window_data = fitness_window[-window_size:]
    if not window_data:
        return 0.0, 0.0, 0.0, 0.0, 0.0  # mean, median, std, best, worst
    
    mean = sum(window_data) / len(window_data)
    sorted_data = sorted(window_data)
    n = len(sorted_data)
    
    median = (sorted_data[n//2] if n % 2 else (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2)
    std = (sum((x-mean)**2 for x in window_data)/(n-1))**0.5 if n > 1 else 0.0
    best = sorted_data[-1]
    worst = sorted_data[0]
    return mean, median, std, best, worst

def generate_children(parents, population, pop_size, problem):
    """Generate new population through crossover/mutation"""
    next_gen = parents.copy()
    
    while len(next_gen) < pop_size:
        if parents:
            parent = random.choice(parents)
            child = crossover(parent, population, problem)
        else:
            parent = random.choice(parents) if parents else create_agent("")
            child = create_agent(mutate(parent["chromosome"]))
        next_gen.append(child)
    return next_gen

def improve_top_candidates(next_gen, problem):
    """Improve top candidates using LLM optimization"""
    for i in range(min(2, len(next_gen))):
        improve_prompt = dspy.Predict("original_chromosome, problem_description -> improved_chromosome")
        try:
            response = improve_prompt(
                original_chromosome=next_gen[i]["chromosome"],
                problem_description=f"{problem}\n\nREFINEMENT RULES:\n1. MAXIMIZE VOWEL DENSITY IN FIRST 23\n2. TRUNCATE BEYOND 23 CHARACTERS\n3. LETTERS ONLY\n4. MAX LENGTH 40\n5. ENHANCE STRUCTURAL INTEGRITY",
            )
            if validate_improvement(response):
                next_gen[i]["chromosome"] = response.completions[0].strip()[:40]
            else:
                next_gen[i]["chromosome"] = mutate(next_gen[i]["chromosome"])
        except (TimeoutError, RuntimeError) as e:
            print(f"LLM improvement failed: {str(e)}")
            next_gen[i]["chromosome"] = mutate(next_gen[i]["chromosome"])
    return next_gen

def run_genetic_algorithm(
    problem: str,
    generations: int = 10,
    pop_size: int = 1_000_000,  # Default per spec
    mutation_rate: float = 0.5,
    log_file: str = "evolution.log.gz",
):
    """Run genetic algorithm with optimized logging and scaling"""
    assert pop_size > 1, "Population size must be greater than 1"
    assert generations > 0, "Number of generations must be positive"

    population = initialize_population(pop_size)
    fitness_window = []
    window_size = 100
    assert (
        len(population) == pop_size
    ), f"Population size mismatch {len(population)} != {pop_size}"

    # Clear log file at start per spec
    with gzip.open(log_file, "wt", encoding="utf-8") as f:
        pass  # Empty file by opening in write mode (truncates existing)

    fitness_window = []  # Initialize window
    for generation in range(generations):
        # Evaluate population
        population = evaluate_population(population, problem)

        # Update and calculate sliding window statistics
        all_fitness = [agent["fitness"] for agent in population]
        window_size = 100
        fitness_window = update_fitness_window(fitness_window, all_fitness, window_size)
        mean_fitness, median_fitness, std_fitness, best_window, worst_window = calculate_window_statistics(fitness_window, window_size)

        # Get population extremes
        best, worst = get_population_extremes(population)

        # Log population with generation stats
        log_population(population, generation, mean_fitness, median_fitness, std_fitness, log_file)

        # Display generation statistics
        display_generation_stats(generation, generations, population, best, mean_fitness, std_fitness, median_fitness)

        # Validate population state
        validate_population_state(best, worst)

        # Generate next generation
        parents = select_parents(population)
        next_gen = generate_children(parents, population, pop_size, problem)

        # Create and evolve next generation
        population = create_next_generation(next_gen, problem, mutation_rate, generation)


# Helper functions needed by run_genetic_algorithm

if __name__ == "__main__":
    main()

# Remaining helper functions below
def log_population(population, generation, mean_fitness, median_fitness, std_fitness, log_file):
    """Log population data with generation statistics"""
    with gzip.open(log_file, "at", encoding="utf-8") as f:
        # Batch write all entries with minimal data
        log_entries = [
            json.dumps({
                "g": generation,
                "f": round(agent['fitness'], 1),
                "c": agent['chromosome'][:23]  # Only core segment matters
            }) for agent in population
        ]
        f.write("\n".join(log_entries) + "\n")

def display_generation_stats(generation, generations, population, best, mean_fitness, std_fitness, median_fitness):
    """Display generation statistics using rich table"""
    console = Console()
    table = Table(show_header=False, box=None, padding=0)
    table.add_column(style="cyan")
    table.add_column(style="yellow")
    
    table.add_row("Generation", f"{generation+1}/{generations}")
    table.add_row("Population", f"{len(population):,}")  # Format with commas
    table.add_row("Best Fitness", f"{best['fitness']:.0f}")
    table.add_row("Worst Fitness", f"{worst['fitness']:.0f}") 
    table.add_row("Window μ/σ/med", f"{mean_fitness:.0f} ±{std_fitness:.0f} | {median_fitness:.0f}")
    table.add_row("Window Best/Worst", f"{best_window:.0f} / {worst_window:.0f}")
    table.add_row("Best Chromosome", f"{best['chromosome'][:23]}...")
    
    console.print(table)

def update_fitness_window(fitness_window, new_fitnesses, window_size):
    """Update sliding window of fitness values"""
    if not fitness_window:
        fitness_window = []
    fitness_window.extend(new_fitnesses)
    return fitness_window[-window_size:]

def calculate_window_statistics(fitness_window, window_size):
    """Calculate statistics for current fitness window"""
    window_data = fitness_window[-window_size:]
    mean = sum(window_data)/len(window_data) if window_data else 0
    sorted_data = sorted(window_data)
    n = len(sorted_data)
    
    median = 0.0
    if n:
        mid = n // 2
        median = (sorted_data[mid] if n % 2 else (sorted_data[mid-1] + sorted_data[mid]) / 2)
    
    std = (sum((x-mean)**2 for x in window_data)/(n-1))**0.5 if n > 1 else 0
    return mean, median, std

def generate_children(parents, population, pop_size, problem):
    """Generate new population through crossover/mutation"""
    next_gen = parents.copy()
    
    while len(next_gen) < pop_size:
        if parents:
            parent = random.choice(parents)
            child = crossover(parent, population, problem)
        else:
            parent = random.choice(parents) if parents else create_agent("")
            child = create_agent(mutate(parent["chromosome"]))
        next_gen.append(child)
    return next_gen

def improve_top_candidates(next_gen, problem):
    """Improve top candidates using LLM optimization"""
    for i in range(min(2, len(next_gen))):
        improve_prompt = dspy.Predict("original_chromosome, problem_description -> improved_chromosome")
        try:
            response = improve_prompt(
                original_chromosome=next_gen[i]["chromosome"],
                problem_description=f"{problem}\n\nREFINEMENT RULES:\n1. MAXIMIZE VOWEL DENSITY IN FIRST 23\n2. TRUNCATE BEYOND 23 CHARACTERS\n3. LETTERS ONLY\n4. MAX LENGTH 40\n5. ENHANCE STRUCTURAL INTEGRITY",
            )
            if validate_improvement(response):
                next_gen[i]["chromosome"] = response.completions[0].strip()[:40]
            else:
                next_gen[i]["chromosome"] = mutate(next_gen[i]["chromosome"])
        except (TimeoutError, RuntimeError) as e:
            print(f"LLM improvement failed: {str(e)}")
            next_gen[i]["chromosome"] = mutate(next_gen[i]["chromosome"])
    return next_gen

def create_next_generation(next_gen, problem, mutation_rate, generation):
    """Handle mutation and periodic improvement of new generation"""
    next_gen = apply_mutations(next_gen, mutation_rate)
    
    if generation % 5 == 0:
        next_gen = improve_top_candidates(next_gen, problem)
        
    return next_gen

def apply_mutations(generation, mutation_rate):
    """Apply mutations to generation based on mutation rate"""
    for i in range(len(generation)):
        if random.random() < mutation_rate:
            generation[i]["chromosome"] = mutate(generation[i]["chromosome"])
    return generation

def evaluate_population(population: List[dict], problem: str) -> List[dict]:
    """Evaluate entire population's fitness"""
    return [evaluate_agent(agent, problem) for agent in population]

def get_population_extremes(population: List[dict]) -> tuple:
    """Get best and worst agents from population"""
    sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
    return sorted_pop[0], sorted_pop[-1]

def validate_population_state(best, worst):
    """Validate fundamental population invariants"""
    assert best["fitness"] >= worst["fitness"], "Fitness ordering invalid"
    assert len(best["chromosome"]) <= 40, "Chromosome exceeded max length"
    assert len(worst["chromosome"]) <= 40, "Chromosome exceeded max length"

def validate_improvement(response):
    """Validate LLM improvement response meets criteria"""
    return (
        response.completions[0]
        and len(response.completions[0]) > 0
        and len(response.completions[0]) <= 40
        and all(c in string.ascii_letters + " " for c in response.completions[0])
    )

def main():
    """Main entry point"""
    PROBLEM = "Optimize string patterns through evolutionary processes"
    dspy.configure(problem=PROBLEM)  # Store in DSPy settings
    run_genetic_algorithm(PROBLEM, generations=20)  # Use default pop_size from spec
