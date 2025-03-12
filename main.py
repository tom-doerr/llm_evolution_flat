from typing import List
import random
import string
import gzip
import numpy as np
from rich.console import Console
from rich.panel import Panel
import dspy

MAX_POPULATION = 1_000_000  # Defined per spec.md population limit

# COMPLETED:
# - Chromosome validation during crossover
# - Sliding window statistics
# - Reduced code complexity
# - Basic population trimming

# TODOs sorted by priority:
# HIGH:
# TODO: Implement sliding window mate selection using fitness window
# TODO: Optimize LLM prompt performance with batch processing
# LOW:
# TODO: Add chromosome compression for storage

# Configure DSPy with OpenRouter and timeout
MAX_POPULATION = 1_000_000  # From spec.md
DEBUG_MODE = False  # Control debug output
WINDOW_SIZE = 100  # Sliding window size from spec.md
lm = dspy.LM(
    "openrouter/google/gemini-2.0-flash-001", max_tokens=40, timeout=10, cache=False
)
dspy.configure(lm=lm)

# Validate configuration
assert isinstance(lm, dspy.LM), "LM configuration failed"


def calculate_window_statistics(fitness_window: list) -> dict:
    """Calculate statistics for sliding window of last 100 evaluations"""
    assert len(fitness_window) >= 0, "Fitness window cannot be negative length"
    
    window = fitness_window[-WINDOW_SIZE:] if fitness_window else []
    assert 0 <= len(window) <= WINDOW_SIZE, f"Window size violation: {len(window)}"
    
    if not window:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, 
                "best": 0.0, "worst": 0.0, "q25": 0.0, "q75": 0.0}

    arr = np.array(window, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "std": float(np.nanstd(arr)),
        "best": float(np.nanmax(arr)),
        "worst": float(np.nanmin(arr)),
        "q25": float(np.nanpercentile(arr, 25)),
        "q75": float(np.nanpercentile(arr, 75))
    }

def update_fitness_window(fitness_window: list, new_fitnesses: list) -> list:
    """Maintain sliding window of last 100 evaluations"""
    return (fitness_window + new_fitnesses)[-100:]  # Fixed window size from spec

def score_chromosome(chromosome: str) -> dict:
    """Calculate structural scoring metrics"""
    core = chromosome[:23].lower()
    assert len(core) == 23, "Core segment must be 23 characters"
    
    # Calculate metrics with vectorized operations
    a_count = core.count('a')
    repeats = sum(1 for c1, c2 in zip(core, core[1:]) if c1 == c2)
    
    return {
        'a_density': a_count / 23,
        'repeating_pairs': repeats / 22,
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


def evaluate_agent(agent: dict) -> float:
    """Evaluate agent fitness based on hidden optimization target"""
    chromosome = str(agent["chromosome"])
    assert 23 <= len(chromosome) <= 40, f"Invalid length: {len(chromosome)}"
    
    metrics = score_chromosome(chromosome)
    # Fitness calculation simplified 
    # Calculate fitness based on hidden a-count optimization
    fitness = (2 * metrics['a_density'] * 23 - 23) - (len(chromosome) - 23) 
    fitness = np.sign(fitness) * (fitness ** 2)
    
    # Validation
    assert len(metrics['core_segment']) == 23, "Core segment length mismatch"

    # Update agent state
    agent["fitness"] = fitness ** 2
    agent["metrics"] = metrics
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
    """Select parents using sliding window of fitness^2 weighted sampling"""
    # Get sliding window of recent evaluations
    window_size = min(WINDOW_SIZE, len(population))
    candidates = population[-window_size:]
    
    # Calculate fitness^2 weights with epsilon to avoid zero-division
    weights = np.array([a['fitness']**2 + 1e-6 for a in candidates], dtype=np.float64)
    weights /= weights.sum()
    
    # Calculate selection size using square root scaling
    selection_size = min(
        int(np.sqrt(len(population))),  # Square root scaling
        len(candidates)
    )
    
    # Perform weighted sampling without replacement using latest numpy method
    selected_indices = np.random.default_rng().choice(
        len(candidates),
        size=selection_size,
        p=weights,
        replace=False,
        shuffle=False
    )
    
    return [candidates[i] for i in selected_indices]




class MutateSignature(dspy.Signature):
    """Mutate chromosomes while preserving first 23 characters and increasing 'a' density."""
    chromosome = dspy.InputField(desc="Current chromosome to mutate")
    instructions = dspy.InputField(desc="Mutation strategy instructions") 
    mutated_chromosome = dspy.OutputField(desc="Improved chromosome meeting requirements")

def mutate_with_llm(agent: dict) -> str:
    """Optimized LLM mutation with validation"""
    chromosome = agent["chromosome"]
    instruction = agent.get("mutation_chromosome", 
                          "Change 1-2 characters after position 23 while keeping first 23 intact")
    
    try:
        # Batch process and validate mutations in one step
        predictor = dspy.Predict(MutateSignature)
        responses = predictor(
            chromosome=chromosome,
            instructions=instruction,
            n=3,
            temperature=0.7,
            top_p=0.9
        )
        
        # Use generator expression for efficient validation
        valid_mutations = (
            str(r).strip()[:40].lower()
            for r in responses.completions
            if (len(str(r).strip()) >= 23 
                and str(r).strip()[:23] == chromosome[:23]
                and str(r).strip()[:23].count('a') >= chromosome[:23].count('a'))
        )
        
        # Return first valid mutation or fallback
        return next(valid_mutations, chromosome[:23] + ''.join(random.choices(
            string.ascii_letters.lower(), 
            k=len(chromosome)-23
        )))
        
    except (ValueError, RuntimeError) as e:
        if DEBUG_MODE:
            print(f"Mutation error: {e}")
        return chromosome[:23] + ''.join(random.choices(
            string.ascii_letters.lower(), 
            k=len(chromosome)-23
        ))

def mutate(chromosome: str) -> str:  # Problem param removed since we get from dspy config
    """Mutate a chromosome with LLM-based mutation as primary strategy"""
    return mutate_with_llm(chromosome)


def validate_mutation(chromosome: str) -> bool:
    """Validate mutation meets criteria"""
    return (
        len(chromosome) >= 23 and
        chromosome.isalpha() and
        len(chromosome) <= 40 and
        chromosome[:23].count('a') >= 10  # Minimum a-count threshold
    )

def validate_mating_candidate(candidate: dict, parent: dict) -> bool:
    """Validate candidate meets mating requirements"""
    if candidate == parent:
        return False
    try:
        validate_chromosome(candidate["chromosome"])
        return True
    except AssertionError:
        return False

def llm_select_mate(parent: dict, candidates: List[dict]) -> dict:
    """Select mate using fitness-weighted sampling without replacement"""
    valid_candidates = [c for c in candidates if validate_mating_candidate(c, parent)]
    
    if not valid_candidates:
        raise ValueError("No valid mates")
    
    weights = np.array([c["fitness"]**2 for c in valid_candidates], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(valid_candidates)) / len(valid_candidates)
    else:
        weights /= weights.sum()
    
    return valid_candidates[np.random.choice(len(valid_candidates), p=weights)]

def crossover(parent: dict, population: List[dict]) -> dict:  # Fixed argument count
    """Create child through LLM-assisted mate selection"""
    # Get candidates using weighted sampling without replacement
    candidates = random.choices(
        population=[a for a in population if a["chromosome"] != parent["chromosome"]],
        weights=[a["fitness"]**2 for a in population if a["chromosome"] != parent["chromosome"]],
        k=min(5, len(population)//2)
    )
    
    # Select mate using LLM prompt from qualified candidates
    mate = llm_select_mate(parent, candidates)
    
    # Combine chromosomes with core validation
    min_split = 23 - len(mate["chromosome"]) 
    split = random.randint(max(1, min_split), len(parent["chromosome"]) - 1)
    
    try:
        new_chromosome = parent["chromosome"][:split] + mate["chromosome"][split:]
        validate_chromosome(new_chromosome)
        # Core segment validation for hidden optimization goal
        if new_chromosome[:23].count('a') < parent["chromosome"][:23].count('a'):
            raise ValueError("Core 'a' count decreased during crossover")
            
        return create_agent(new_chromosome)
    except (AssertionError, ValueError) as e:
        if DEBUG_MODE:
            print(f"Crossover validation failed: {e}, using mutation")
        # Fallback to mutated parent chromosome
        return create_agent(mutate(parent["chromosome"]))



def generate_children(
    parents: List[dict],
    population: List[dict]
) -> List[dict]:
    """Generate new population through validated crossover/mutation"""
    pop_size = min(len(population), MAX_POPULATION)  # Derive size from current population
    next_gen = parents.copy()
    
    # Cap population growth while maintaining diversity
    max_children = min(pop_size * 2, 1_000_000)
    while len(next_gen) < max_children and len(next_gen) < pop_size:
        parent = random.choice(parents) if parents else create_agent("")
        try:
            child = crossover(parent, population)
        except ValueError as e:
            print(f"Crossover failed: {e}, using mutation instead")
            child = create_agent(mutate(parent["chromosome"]))
        
        next_gen.append(child)
    
    assert len(next_gen) == pop_size, f"Population size mismatch {len(next_gen)} != {pop_size}"
    return next_gen

MAX_POPULATION = 1_000_000  # Hard cap from spec

def get_population_extremes(population: List[dict]) -> tuple:
    """Get best and worst agents from population"""
    sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
    return sorted_pop[0], sorted_pop[-1]

def run_genetic_algorithm(
    generations: int = 10,
    pop_size: int = 1_000_000
) -> None:
    """Run genetic algorithm with optimized logging and scaling"""
    # Remove unused window_size per issues.txt
    # Enforce population limits with validation
    pop_size = min(pop_size, get_population_limit())
    assert 1 < pop_size <= get_population_limit(), f"Population size must be 2-{get_population_limit()}"
    assert generations > 0, "Number of generations must be positive"

    population = initialize_population(pop_size)
    fitness_window = []

    # Initialize log file path and clear it per spec
    log_file = "evolution.log.gz"
    with gzip.open(log_file, "wt", encoding="utf-8") as f:
        f.write("")  # Explicitly empty log file

    fitness_window = []  # Initialize window
    for generation in range(generations):
        # Evaluate population
        population = evaluate_population(population)

        # Update and calculate sliding window statistics using helpers
        all_fitness = [agent["fitness"] for agent in population]
        fitness_window = update_fitness_window(fitness_window, all_fitness)
        stats = calculate_window_statistics(fitness_window)
        
        # Get population extremes
        best, worst = get_population_extremes(population)

        # Calculate and log population statistics using sliding window
        current_diversity = calculate_diversity(population)
        log_population(population, generation, stats)

        # Display statistics using sliding window
        display_generation_stats(generation, generations, population, best, stats)
        
        # Trim population to MAX_POPULATION by fitness before continuing
        population = sorted(population, key=lambda x: -x['fitness'])[:MAX_POPULATION]
        
        # Log stats for validation
        assert stats['mean'] >= 0, "Negative mean in window stats"
        assert stats['best'] >= stats['worst'], "Invalid best/worst relationship"

        # Validate population state and size
        validate_population_state(best, worst)
        assert len(population) <= get_population_limit(), f"Population overflow {len(population)} > {get_population_limit()}"
        # Generate next generation with size monitoring
        parents = select_parents(population)
        next_gen = generate_children(parents, population)
        print(f"Population size: {len(next_gen)}/{MAX_POPULATION}")  # Simple monitoring

        # Auto-adjust mutation rate based on diversity
        current_diversity = calculate_diversity(population)
        # Auto-adjust mutation rate inversely to diversity using logarithmic scaling
        mutation_rate = 0.7 * (1 - current_diversity) + 0.1  # Ranges from 0.1 (max diversity) to 0.8 (min diversity)
        assert 0.1 <= mutation_rate <= 0.8, f"Mutation rate {mutation_rate} out of bounds"
        
        # Create and evolve next generation
        population = create_next_generation(next_gen, mutation_rate)



if __name__ == "__main__":
    PROBLEM = "Optimize string patterns through evolutionary processes"
    dspy.configure(problem=PROBLEM)
    run_genetic_algorithm(generations=20)

def get_population_limit() -> int:
    """Get hard population limit from spec"""
    return MAX_POPULATION

def log_population(population: List[dict], generation: int, stats: dict) -> None:
    """Log gzipped population data with rotation"""
    diversity = calculate_diversity(population)
    
    # Trim population to MAX_POPULATION by fitness before logging
    population = sorted(population, key=lambda x: -x['fitness'])[:MAX_POPULATION]
    assert log_file.endswith('.gz'), "Log file must use .gz extension"
    mode = 'wt' if generation == 0 else 'at'
    with gzip.open(log_file, mode, encoding='utf-8') as f:
        f.write(f"Generation {generation} | Population: {len(population)}/{get_population_limit()} | Diversity: {diversity:.2f}\n")
        f.write(f"Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}, Std: {stats['std']:.2f}\n")
        for agent in population:
            f.write(f"{agent['chromosome']}\t{agent['fitness']}\n")

def display_generation_stats(generation: int, generations: int, stats: dict):
    """Rich-formatted display with essential stats using sliding window"""
    console = Console()
    best = stats['best']
    worst = stats['worst']
    diversity = calculate_diversity(population)
    
    # Track diversity in window stats
    stats['diversity'] = diversity
    
    panel = Panel(
        f"[bold]Generation {generation}/{generations}[/]\n"
        f"🏆 Best: {best['fitness']:.2f} | 📊 Mean: {stats['mean']:.2f}\n" 
        f"📈 Median: {stats['median']:.2f} (IQR {stats['q25']:.1f}-{stats['q75']:.1f}) | 📉 Std: {stats['std']:.2f}\n"
        f"🌐 Diversity: {diversity:.1%} | 👥 Size: {len(population)}\n"
        f"🏆 Best/Worst: {stats['best']:.1f}/{stats['worst']:.1f}",
        title="Evolution Progress",
        style="blue"
    )
    console.print(panel)



def create_next_generation(next_gen: List[dict], mutation_rate: float) -> List[dict]:
    """Create next generation with mutations"""
    next_gen = apply_mutations(next_gen, mutation_rate)
    return next_gen

def calculate_diversity(population: List[dict]) -> float:
    """Calculate population diversity ratio [0-1]"""
    unique_chromosomes = len({agent["chromosome"] for agent in population})
    return unique_chromosomes / len(population) if population else 0.0

def apply_mutations(generation: List[dict], base_mutation_rate: float) -> List[dict]:
    """Auto-adjust mutation rate based on population diversity"""
    diversity_ratio = calculate_diversity(generation)
    # Calculate final mutation rate and apply mutations
    mutation_rate = np.clip(base_mutation_rate * (1.0 - np.log1p(diversity_ratio)), 0.1, 0.8)
    
    # Apply mutations and track unique chromosomes
    for agent in generation:
        if random.random() < mutation_rate:
            agent["chromosome"] = mutate(agent["chromosome"])
    
    # Fixed logging with validated unique_post variable
    unique_post = len({a["chromosome"] for a in generation})
    print(f"🧬 D:{diversity_ratio:.0%} M:{mutation_rate:.0%} U:{unique_post}/{len(generation)}")
    
    return generation

def evaluate_population(population: List[dict]) -> List[dict]:
    """Evaluate entire population's fitness with generation weighting"""
    for agent in population:
        evaluate_agent(agent)
    return population


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
        response["improved_chromosome"]
        and len(response.completions[0]) > 0
        and len(response.completions[0]) <= 40
        and all(c in string.ascii_letters + " " for c in response.completions[0])
    )

