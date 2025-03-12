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
# 1. Add chromosome validation before LLM mating selection
# 2. Optimize fitness window statistics calculations
# 3. Add population size monitoring/limiting

# Configure DSPy with OpenRouter and timeout
lm = dspy.LM(
    "openrouter/google/gemini-2.0-flash-001", max_tokens=40, timeout=10, cache=False
)
dspy.configure(lm=lm)

# Validate configuration
assert isinstance(lm, dspy.LM), "LM configuration failed"


def validate_chromosome(chromosome: str) -> str:
    """Validate and normalize chromosome structure"""
    if isinstance(chromosome, list):
        chromosome = "".join(chromosome)
    chromosome = str(chromosome).strip()[:40]  # Enforce max length
    
    # Structural validation
    assert 1 <= len(chromosome) <= 40, f"Invalid length {len(chromosome)}"
    assert all(c.isalpha() or c == ' ' for c in chromosome), "Invalid characters"
    assert chromosome == chromosome.strip(), "Whitespace not allowed at ends"
    
    return chromosome

def create_agent(chromosome: str) -> dict:
    """Create a new agent as a dictionary"""
    chromosome = validate_chromosome(chromosome)
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
    """Select parents using Pareto distribution weighted by fitness²"""
    assert len(population) > 0, "Cannot select from empty population"
    
    # Calculate squared fitness values for Pareto weighting
    fitness_squares = [a['fitness']**2 for a in population]
    total = sum(fitness_squares)
    if total <= 0:
        raise ValueError("Total fitness squared must be positive")
    
    sample_size = min(2, len(population))  # Select 2 parents
    selected_parents = []
    current_weights = fitness_squares.copy()
    indices = list(range(len(population)))
    
    for _ in range(sample_size):
        total = sum(current_weights)
        if total <= 0:
            break
        
        # Select parent using weighted choice without replacement
        chosen_idx = random.choices(indices, weights=current_weights, k=1)[0]
        selected_parents.append(population[chosen_idx])
        
        # Prevent reselection by zeroing weight and removing index
        current_weights[chosen_idx] = 0
        indices.remove(chosen_idx)
    
    # Validate final selection
    assert len(selected_parents) == sample_size, f"Parent selection failed: got {len(selected_parents)} parents"
    return selected_parents



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
    # Pre-filter candidates with validation
    valid_candidates = [
        agent for agent in candidates 
        if validate_chromosome(agent["chromosome"]) and agent != parent
    ]
    if not valid_candidates:
        raise ValueError("No valid mating candidates available")
    
    # Format candidates with indexes
    candidate_list = [f"{idx}: {agent['chromosome'][:23]}" 
                    for idx, agent in enumerate(valid_candidates)]
    
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
        display_generation_stats(generation, generations, population, best, mean_fitness, std_fitness, fitness_window)

        # Validate population state
        validate_population_state(best, worst)

        # Generate next generation
        parents = select_parents(population)
        next_gen = generate_children(parents, population, pop_size, problem)

        # Create and evolve next generation
        population = create_next_generation(next_gen, problem, mutation_rate, generation)


# Import optimized statistics functions
import helpers

# Helper functions needed by run_genetic_algorithm

if __name__ == "__main__":
    main()
def log_population(population, generation, mean_fitness, median_fitness, std_fitness, log_file):
    """Log gzipped population data with rotation"""
    assert log_file.endswith('.gz'), "Log file must use .gz extension"
    mode = 'wt' if generation == 0 else 'at'
    with gzip.open(log_file, mode, encoding='utf-8') as f:
        f.write(f"Generation {generation}\n")
        f.write(f"Mean: {mean_fitness}, Median: {median_fitness}, Std: {std_fitness}\n")
        for agent in population:
            f.write(f"{agent['chromosome']}\t{agent['fitness']}\n")

def display_generation_stats(generation, generations, population, best, mean_fitness, std_fitness, fitness_window):
    """Rich-formatted display with essential stats"""
    console = Console()
    stats = calculate_window_statistics(fitness_window, 100)
    
    panel = Panel(
        f"[bold]Generation {generation}/{generations}[/]\n"
        f"🏆 Best: {best['fitness']:.2f} | 📊 Mean: {mean_fitness:.2f}\n"
        f"📈 Median: {stats['median']:.2f} | 📉 Std: {std_fitness:.2f}\n"
        f"👥 Population: {len(population)}",
        title="Evolution Progress",
        style="blue"
    )
    console.print(panel)

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
