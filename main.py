import random
import string
import gzip
from rich.console import Console
from rich.table import Table
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
    """Select parents using Pareto distribution weighted by fitness^2 with deduplication"""
    # Calculate raw fitness values and handle zero-sum case
    fitness_values = [agent["fitness"] for agent in population]
    
    # Square fitness values for Pareto weighting as per spec
    squared_fitness = [f**2 for f in fitness_values]
    total_weight = sum(squared_fitness)
    
    # Fallback to random selection if all weights zero
    if total_weight <= 0:
        return random.sample(population, k=len(population)//2)
    
    # Weighted selection without replacement with deduplication
    selected = []
    unique_chromosomes = set()
    
    # Create candidate pool with deduplication
    candidates = [agent for agent in population if agent["chromosome"] not in unique_chromosomes]
    weights = [squared_fitness[i] for i, agent in enumerate(population) if agent["chromosome"] not in unique_chromosomes]
    
    while len(selected) < len(population)//2 and len(candidates) > 0:
        # Weighted sample without replacement
        chosen = random.choices(candidates, weights=weights, k=1)[0]
        selected.append(chosen)
        unique_chromosomes.add(chosen["chromosome"])
        
        # Remove chosen from candidates
        index = candidates.index(chosen)
        del candidates[index]
        del weights[index]
        
    return selected


def mutate_with_llm(chromosome: str, problem: str) -> str:
    """Mutate chromosome using LLM-based rephrasing"""
    mutate_prompt = dspy.Predict("original_chromosome, problem_description -> mutated_chromosome")
    # Strict mutation with validation
    try:
        response = mutate_prompt(
            original_chromosome=chromosome,
            problem_description=f"{problem}\nRULES:\n1. IMPROVE INITIAL SEGMENT QUALITY\n2. REMOVE UNNECESSARY ELEMENTS\n3. ALPHABETIC CHARACTERS ONLY\n4. STRICT LENGTH LIMITS",  # More obfuscated
        )
        mutated = str(response.mutated_chromosome).strip()[:40]  # Hard truncate
        mutated = ''.join([c if c.isalpha() else random.choice('abcdefghijklmnopqrstuvwxyz') for c in mutated])  # Enforce letters
        mutated = mutated[:23] + ''  # Enforce truncation after 23
        mutated = mutated.ljust(23, 'x')[:40]  # Fill with neutral chars
        
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


def llm_select_mate(parent: dict, candidates: List[dict], problem: str) -> dict:
    """Select mate using LLM prompt with candidate chromosomes"""
    # Format candidates with indexes
    candidate_list = [f"{idx}: {agent['chromosome'][:23]}" 
                    for idx, agent in enumerate(candidates)]
    
    # Create prompt with mating rules
    prompt = dspy.Predict("parent_chromosome, candidates, problem -> best_candidate_id")
    response = prompt(
        parent_chromosome=parent["chromosome"],
        candidates="\n".join(candidate_list),
        problem=f"{problem}\nRULES:\n1. OPTIMIZE INITIAL PORTION\n2. MAINTAIN LENGTH LIMIT\n3. SELECT COMPATIBLE PAIR",
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
    # Get candidates using weighted sampling without replacement with deduplication
    candidates = random.sample(
        [a for a in population if a["chromosome"] != parent["chromosome"]],
        k=min(5, len(population)//2)
    )
    
    # Select mate using LLM prompt from qualified candidates
    mate = llm_select_mate(parent, candidates, problem)
    
    # Combine chromosomes
    split = random.randint(1, len(parent["chromosome"]) - 1)
    new_chromosome = parent["chromosome"][:split] + mate["chromosome"][split:]
    return create_agent(new_chromosome)


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

        # Track sliding window of last 100 evaluations as per spec
        window_size = 100
        all_fitness = [agent["fitness"] for agent in population]
        
        # Maintain window in memory with thread-safe approach
        current_window = getattr(run_genetic_algorithm, "fitness_window", [])
        current_window = (current_window + all_fitness)[-window_size:]
        run_genetic_algorithm.fitness_window = current_window
        
        # Calculate robust statistics
        window_fitness = current_window
        mean_fitness = sum(window_fitness) / len(window_fitness) if window_fitness else 0
        sorted_fitness = sorted(window_fitness)
        # Proper median calculation for even/odd window sizes
        n = len(sorted_fitness)
        if n == 0:
            median_fitness = 0.0
        else:
            mid = n // 2
            if n % 2 == 1:
                median_fitness = sorted_fitness[mid]
            else:
                median_fitness = (sorted_fitness[mid-1] + sorted_fitness[mid]) / 2
        std_fitness = (sum((x-mean_fitness)**2 for x in window_fitness)/len(window_fitness))**0.5 if window_fitness else 0

        # Get best/worst before logging
        sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
        best = sorted_pop[0]
        worst = sorted_pop[-1]

        # Detailed compressed logging per spec
        with gzip.open(log_file, "at", encoding="utf-8") as f:
            for agent in population:
                f.write(f"{generation},{agent['fitness']},{agent['chromosome']}\n")

        # Rich formatted output
        console = Console()
        table = Table(show_header=False, box=None, padding=0)
        table.add_column(style="cyan")
        table.add_column(style="yellow")
        
        table.add_row("Generation", f"{generation+1}/{generations}")
        table.add_row("Population", f"{pop_size} agents")
        table.add_row("Best", f"{best['chromosome'][:23]}... [green]({best['fitness']:.1f})")
        table.add_row("Worst", f"{worst['chromosome'][:23]}... [red]({worst['fitness']:.1f})")
        table.add_row("Stats", f"μ={mean_fitness:.1f} ±{std_fitness:.1f} | Med={median_fitness:.1f}")
        
        console.print(table)

        # Validate population state
        assert best["fitness"] >= worst["fitness"], "Fitness ordering invalid"
        assert len(best["chromosome"]) <= 40, "Chromosome exceeded max length"
        assert len(worst["chromosome"]) <= 40, "Chromosome exceeded max length"

        # Select parents and create next generation
        parents = select_parents(population)
        next_gen = parents.copy()

        # Create children through crossover with parent validation
        while len(next_gen) < pop_size:
            if parents:
                parent = random.choice(parents)
                child = crossover(parent, population, problem)
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
                # LLM-based improvement with strict rules per spec
                improve_prompt = dspy.Predict(
                    "original_chromosome, problem_description -> improved_chromosome"
                )
                try:
                    response = improve_prompt(
                        original_chromosome=next_gen[i]["chromosome"],
                        problem_description=f"{problem}\n\nOPTIMIZATION RULES:\n1. ENHANCE INITIAL PORTION\n2. STRICT LENGTH LIMITS\n3. ALPHABETIC CHARACTERS ONLY\n4. NO METACOMMENTARY",
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
    PROBLEM = "Optimize string composition according to evolving fitness criteria"  # Obfuscated per spec
    run_genetic_algorithm(PROBLEM, generations=20, pop_size=10)
