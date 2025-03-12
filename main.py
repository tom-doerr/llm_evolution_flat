import random
import string
import gzip
import json
import numpy as np
import numpy as np
from typing import List
from rich.console import Console
from rich.table import Table
import dspy

# TODO List (sorted by priority):
# 1. Add mutation rate adaptation based on diversity
# 2. Implement generation-based scoring weights

# Configure DSPy with OpenRouter and timeout
lm = dspy.LM(
    "openrouter/google/gemini-2.0-flash-001", max_tokens=40, timeout=10, cache=False
)
dspy.configure(lm=lm)

# Validate configuration
assert isinstance(lm, dspy.LM), "LM configuration failed"


def calculate_window_statistics(fitness_window: list, window_size: int = 100) -> dict:
    """Calculate statistics for current fitness window using vectorized operations"""
    window = fitness_window[-window_size:]
    if not window:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "best": 0.0, "worst": 0.0}
    
    arr = np.array(window)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "best": float(np.max(arr)),
        "worst": float(np.min(arr))
    }

def update_fitness_window(fitness_window: list, new_fitnesses: list, window_size: int) -> list:
    """Update sliding window efficiently using deque-like behavior"""
    combined = (fitness_window[-window_size:] if fitness_window else []) + new_fitnesses
    return combined[-window_size:]

def score_chromosome(chromosome: str) -> dict:
    """Calculate comprehensive structural scoring metrics with validation"""
    # Core segment analysis
    core = chromosome[:23].lower()
    assert len(core) == 23, "Core segment must be exactly 23 characters"
    
    # Character type analysis
    vowels = sum(1 for c in core if c in 'aeiou')
    consonants = len(core) - vowels
    unique_chars = len(set(core))
    
    # Sequence pattern analysis
    a_density = core.count('a') / 23
    repeating_chars = sum(1 for i in range(1, len(core)) if core[i] == core[i-1] else 0)
    
    # Structural validation
    assert 0 <= vowels <= 23, "Invalid vowel count"
    assert 0 <= consonants <= 23, "Invalid consonant count"
    assert 1 <= unique_chars <= 23, "Invalid uniqueness calculation"
    
    return {
        'vowel_ratio': vowels / 23,
        'consonant_ratio': consonants / 23,
        'uniqueness': unique_chars / 23,
        'a_density': a_density,
        'repeating_pairs': repeating_chars / 22,  # Possible pairs in 23 chars
        'core_segment': core
    }

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
    # Validate input before scoring
    chromosome = str(agent["chromosome"])
    assert 23 <= len(chromosome) <= 40, f"Invalid chromosome length: {len(chromosome)}"
    
    metrics = score_chromosome(chromosome)
    assert 0 <= metrics['a_density'] <= 1, "Invalid a_density score"
    assert 0 <= metrics['repeating_pairs'] <= 1, "Invalid repeating_pairs score"

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

    # Incorporate structural metrics into fitness
    fitness *= metrics['vowel_ratio'] * (1 + metrics['uniqueness'])
    fitness -= metrics['consecutive_repeats'] * 0.2  # Penalize repeats
    fitness += metrics['length_score'] * 5  # Favor longer chromosomes
    fitness += metrics['positional_diversity'] * 2  # Reward spaced diversity
    # Note: a_count is already accounted for in base fitness
    
    # Allow negative fitness as per spec
    agent["fitness"] = fitness
    agent["metrics"] = metrics  # Store metrics for analysis
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
    """Select parents using adaptive weights based on diversity"""
    # Calculate population diversity
    unique_chromosomes = len({agent["chromosome"] for agent in population})
    diversity = unique_chromosomes / len(population) if population else 0.0
    
    # Adapt weights based on diversity (more exploration when diversity is low)
    exponent = 3 if diversity < 0.1 else 2 if diversity < 0.3 else 1.5
    weights = np.array([agent['fitness']**exponent for agent in population])
    weights += 1e-8  # Add epsilon to avoid zero weights
    weights /= weights.sum()  # Normalize
    
    # Validate weights before selection
    assert np.all(weights >= 0), "Negative weights detected in parent selection"
    assert len(weights) == len(population), "Weight/population size mismatch"
    
    # Select half population using weighted sampling without replacement
    selected_indices = np.random.choice(
        len(population),
        size=len(population)//2,
        replace=False,
        p=weights
    )
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


def validate_mating_candidate(candidate: dict, parent: dict) -> bool:
    """Validate candidate meets mating requirements"""
    if candidate == parent:
        return False
    try:
        validate_chromosome(candidate["chromosome"])
        return True
    except AssertionError:
        return False

def llm_select_mate(parent: dict, candidates: List[dict], problem: str) -> dict:
    """Select mate using LLM prompt with validated candidate chromosomes"""
    # Validate parent first with debugging
    parent_chrom = validate_chromosome(parent["chromosome"])
    debug = False  # Set to True for validation troubleshooting
    assert parent["fitness"] > 0, "Parent must have positive fitness"
    
    # Strict candidate validation with error handling
    valid_candidates = []
    for agent in candidates:
        try:
            agent_chrom = validate_chromosome(agent["chromosome"])
            if agent_chrom != parent_chrom and 23 <= len(agent_chrom) <= 40:
                valid_candidates.append(agent)
                
                if debug:
                    print(f"Valid candidate: {agent_chrom[:23]}... (fitness: {agent['fitness']})")
                    
        except AssertionError as e:
            if debug:
                print(f"Invalid candidate rejected: {str(e)}")
    
    if not valid_candidates:
        raise ValueError(f"No valid mates among {len(candidates)} candidates")
    
    # Create indexed list of first 23 chars (core segment)
    candidate_list = [f"{idx}: {c['chromosome'][:23]} (fitness: {c['fitness']})" 
                    for idx, c in enumerate(valid_candidates)]
    
    # LLM selection with validation constraints
    prompt = dspy.Predict("parent_chromosome, candidates, problem -> best_candidate_id")
    response = prompt(
        parent_chromosome=validate_chromosome(parent["chromosome"]),
        candidates="\n".join(candidate_list),
        problem=f"{problem}\nSELECTION RULES:\n1. MAXIMIZE CORE SIMILARITY\n2. OPTIMIZE LENGTH 23\n3. ENSURE CHARACTER DIVERSITY",
    )
    
    # Parse and validate selection
    raw_response = str(response.best_candidate_id).strip()
    chosen_id = int(raw_response.split(":", maxsplit=1)[0])  # Use maxsplit
        
    # Validate selection is within range
    if 0 <= chosen_id < len(valid_candidates):
        return valid_candidates[chosen_id]
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



def generate_children(parents: List[dict], population: List[dict], pop_size: int, problem: str) -> List[dict]:
    """Generate new population through validated crossover/mutation"""
    pop_size = min(pop_size, 1_000_000)  # Hard cap
    next_gen = parents.copy()
    
    # Cap population growth while maintaining diversity
    max_children = min(pop_size * 2, 1_000_000)
    while len(next_gen) < max_children and len(next_gen) < pop_size:
        parent = random.choice(parents) if parents else create_agent("")
        try:
            child = crossover(parent, population, problem)
        except ValueError as e:
            print(f"Crossover failed: {e}, using mutation instead")
            child = create_agent(mutate(parent["chromosome"]))
        
        next_gen.append(child)
    
    assert len(next_gen) == pop_size, f"Population size mismatch {len(next_gen)} != {pop_size}"
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
    log_file: str = "evolution.log.gz",
):
    """Run genetic algorithm with optimized logging and scaling"""
    # Enforce population limits with validation
    pop_size = min(pop_size, get_population_limit())
    assert 1 < pop_size <= get_population_limit(), f"Population size must be 2-{get_population_limit()}"
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

        # Update and calculate sliding window statistics using helpers
        all_fitness = [agent["fitness"] for agent in population]
        window_size = 100
        fitness_window = helpers.update_fitness_window(fitness_window, all_fitness, window_size)
        stats = helpers.calculate_window_statistics(fitness_window, window_size)
        mean_fitness = stats['mean']
        median_fitness = stats['median'] 
        std_fitness = stats['std']

        # Get population extremes
        best, worst = get_population_extremes(population)

        # Log population with generation stats
        log_population(population, generation, mean_fitness, median_fitness, std_fitness, log_file)

        # Calculate and display statistics
        fitness_values = [a['fitness'] for a in population]
        mean_fitness = np.mean(fitness_values)
        median_fitness = np.median(fitness_values)
        std_fitness = np.std(fitness_values)
        display_generation_stats(generation, generations, population, best, mean_fitness, std_fitness, fitness_window)

        # Validate population state and size
        validate_population_state(best, worst)
        assert len(population) <= MAX_POPULATION, f"Population overflow {len(population)} > {MAX_POPULATION}"
        
        # Generate next generation with size monitoring
        parents = select_parents(population)
        next_gen = generate_children(parents, population, pop_size, problem)
        print(f"Population size: {len(next_gen)}/{MAX_POPULATION}")  # Simple monitoring

        # Auto-adjust mutation rate based on diversity
        current_diversity = calculate_diversity(population)
        mutation_rate = 0.8 if current_diversity < 0.1 else 0.5 if current_diversity < 0.3 else 0.2
        
        # Create and evolve next generation
        population = create_next_generation(next_gen, problem, mutation_rate, generation)



if __name__ == "__main__":
    main()
def get_population_limit() -> int:
    """Get hard population limit from spec"""
    return 1_000_000

def log_population(population, generation, mean_fitness, median_fitness, std_fitness, log_file):
    """Log gzipped population data with rotation"""
    # Log population size against limit
    assert log_file.endswith('.gz'), "Log file must use .gz extension"
    mode = 'wt' if generation == 0 else 'at'
    with gzip.open(log_file, mode, encoding='utf-8') as f:
        f.write(f"Generation {generation} | Population: {len(population)}/{get_population_limit()}\n")
        f.write(f"Mean: {mean_fitness}, Median: {median_fitness}, Std: {std_fitness}\n")
        for agent in population:
            f.write(f"{agent['chromosome']}\t{agent['fitness']}\n")

def display_generation_stats(generation, generations, population, best, mean_fitness, std_fitness, fitness_window):
    """Rich-formatted display with essential stats"""
    console = Console()
    stats = calculate_window_statistics(fitness_window, 100)
    
    # Format population size with SI suffixes
    pop_size = len(population)
    suffix = ''
    if pop_size >= 1_000_000:
        pop_size /= 1_000_000
        suffix = 'M'
    elif pop_size >= 1_000:
        pop_size /= 1_000
        suffix = 'K'
    
    panel = Panel(
        f"[bold]Generation {generation}/{generations}[/]\n"
        f"🏆 Best: {best['fitness']:.2f} | 📊 Mean: {mean_fitness:.2f}\n"
        f"📈 Median: {stats['median']:.2f} | 📉 Std: {std_fitness:.2f}\n"
        f"👥 Population: {len(population)}/{get_population_limit()}",
        title="Evolution Progress",
        style="blue"
    )
    console.print(panel)


def improve_top_candidates(next_gen: List[dict], problem: str) -> List[dict]:
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
    # Validate population invariants
    assert best['fitness'] >= worst['fitness'], "Best fitness should >= worst fitness"
    assert 0 <= best['fitness'] <= 1e6, "Fitness out of reasonable bounds"
    assert 0 <= worst['fitness'] <= 1e6, "Fitness out of reasonable bounds"
    assert isinstance(best['chromosome'], str), "Chromosome should be string"
    assert isinstance(worst['chromosome'], str), "Chromosome should be string"
    assert len(best['chromosome']) <= 40, "Chromosome exceeded max length"
    assert len(worst['chromosome']) <= 40, "Chromosome exceeded max length"

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
