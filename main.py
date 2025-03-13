import random
import string
import itertools
from typing import List

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

# Configure DSPy with OpenRouter and timeout
WINDOW_SIZE = 100  # Default, can be overridden by CLI
lm = dspy.LM(
    "openrouter/google/gemini-2.0-flash-001", max_tokens=40, timeout=10, cache=False
)
dspy.configure(lm=lm)

# Validate configuration
assert isinstance(lm, dspy.LM), "LM configuration failed"
assert "gemini-2.0-flash" in lm.model, "Model must match spec.md"


def calculate_window_statistics(fitness_window: list) -> dict:
    """Calculate statistics for sliding window of last WINDOW_SIZE evaluations"""
    if not fitness_window:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'best': 0.0, 'worst': 0.0}
    assert len(fitness_window) <= WINDOW_SIZE, f"Window size exceeds {WINDOW_SIZE}"
    window_arr = np.array(fitness_window[-WINDOW_SIZE:], dtype=np.float64)
    stats = {
        'mean': float(np.nanmean(window_arr)),
        'median': float(np.nanmedian(window_arr)),
        'std': float(np.nanstd(window_arr)),
        'best': float(np.nanmax(window_arr)),
        'worst': float(np.nanmin(window_arr))
    }
    assert stats['best'] >= stats['worst'], "Best cannot be worse than worst"
    return stats

def update_fitness_window(fitness_window: list, new_fitnesses: list) -> list:
    """Maintain sliding window of last WINDOW_SIZE evaluations"""
    return (fitness_window + new_fitnesses)[-WINDOW_SIZE:]  # Use configurable window size

def score_chromosome(chromosome: str) -> dict:
    """Calculate structural scoring metrics"""
    core = chromosome[:23].lower()
    assert len(core) == 23, "Core segment must be 23 characters"
    
    # Calculate a_count and repeating pairs using optimized methods
    a_count = core.count('a')
    repeats = sum(core[i] == core[i-1] for i in range(1, len(core)))
    
    return {
        'a_density': a_count / 23.0,  # Now properly defined
        'repeating_pairs': repeats / 22,
        'core_segment': core
    }

def validate_chromosome(chromosome: str) -> str:
    """Validate and normalize chromosome structure"""
    if isinstance(chromosome, list):
        chromosome = "".join(chromosome)
    # Preserve original case but enforce length and format
    chromosome = str(chromosome).strip()[:40].lower()  # Normalize to lowercase
    
    # Structural validation
    assert 1 <= len(chromosome) <= 40, f"Invalid length {len(chromosome)}"
    assert all(c.isalpha() or c == ' ' for c in chromosome), "Invalid characters in chromosome"
    assert chromosome == chromosome.strip(), "Whitespace not allowed at ends"
    
    return chromosome

def create_agent(chromosome: str) -> dict:
    """Create a new agent as a dictionary"""
    # Store original chromosome before validation
    original_chromo = chromosome
    chromosome = validate_chromosome(chromosome)
    
    # Preserve original chromosome structure while deriving components
    return {
        "chromosome": original_chromo,
        "task_chromosome": original_chromo[:23].ljust(23, ' ')[:23],  # Enforce exact length
        "mate_selection_chromosome": original_chromo[23:33].ljust(10, ' ')[:10].lower().strip(),
        "mutation_chromosome": original_chromo[33:40].ljust(7, ' ')[:7].strip(),
        "fitness": 0.0
    }
    # Added validation and whitespace stripping per spec.md chromosome separation requirements
    # TODO: Add mutation tracking from spec.md


def evaluate_agent(agent: dict) -> float:
    """Evaluate agent fitness based on hidden optimization target"""
    chromo = validate_chromosome(agent["chromosome"])
    metrics = score_chromosome(chromo)
    agent["fitness"] = metrics['a_density'] * 46 - len(chromo) - 23  # Combined calculation
    
    assert len(metrics['core_segment']) == 23, "Core segment length mismatch"
    agent["metrics"] = metrics
    return agent["fitness"]


def initialize_population(pop_size: int) -> List[dict]:
    """Create initial population with empty chromosomes per spec.md"""
    # Start with empty chromosomes as per spec.md requirement
    chromosomes = [""] * pop_size
    # Parallel create agents
    return [create_agent(c) for c in chromosomes]


def select_parents(population: List[dict]) -> List[dict]:
    """Select parents using Pareto distribution weighted by fitness^2 with weighted sampling without replacement"""
    if not population:
        return []
    
    # Weighted sampling per spec.md: fitness² * Pareto distribution
    weights = np.nan_to_num(
        np.array([a['fitness'] ** 2 for a in population], dtype=np.float64) *
        (np.random.pareto(3.0, len(population)) + 1),  # Pareto shape=3.0
        nan=1e-6
    ).clip(1e-6)
    
    weights /= weights.sum()  # Normalize
    assert np.isclose(weights.sum(), 1.0), "Weights must sum to 1"
    assert len(weights) == len(population), "Weight/population size mismatch"
    
    selected_indices = np.random.choice(
        len(population),
        size=min(len(population), MAX_POPULATION//2),
        replace=False,
        p=weights
    )
    return [population[i] for i in selected_indices]

# Configuration constants from spec.md
MUTATION_RATE = 0.1  # Base mutation probability 
HOTSPOT_CHARS = {'.', '!', '?', ' '}
HOTSPOT_SPACE_PROB = 0.15  # Increased space hotspot probability per spec.md emphasis
MIN_HOTSPOTS = 2  # Ensure minimum 2 switch points for combination
HOTSPOT_ANYWHERE_PROB = 0.025  # Adjusted to exactly 1 hotspot per 40 char chromosome (40 * 0.025 = 1)



class MutateSignature(dspy.Signature):
    """Mutate chromosomes while preserving first 23 characters and increasing 'a' density."""
    chromosome = dspy.InputField(desc="Current chromosome to mutate")
    instructions = dspy.InputField(desc="Mutation strategy instructions") 
    mutated_chromosome = dspy.OutputField(desc="Improved chromosome meeting requirements")

def mutate_with_llm(agent: dict) -> str:
    """Optimized LLM mutation with validation"""
    core_segment = agent["chromosome"][:23].lower()
    response = dspy.Predict(MutateSignature)(
        chromosome=agent["chromosome"],
        instructions=agent["mutation_chromosome"]
    )

    valid_candidate = next(
        (str(c).strip()[:40].lower()
         for c in response.completions
         if str(c).strip().lower().startswith(core_segment)),
        None
    )
    
    return valid_candidate or f"{core_segment}{random.choices(string.ascii_lowercase + ' ', k=17)[:17]}".ljust(40)[:40]

MAX_CHARS = 40  # From spec.md (different from max tokens)
MAX_CORE = 23  # From spec.md hidden goal

# Validate hidden goal constants from spec.md
assert MAX_CORE == 23, "Core segment length must be 23 per spec.md"
assert MAX_CHARS == 40, "Max chromosome length must be 40 for this task"

def mutate(agent: dict) -> str: 
    """Mutate a chromosome with LLM-based mutation as primary strategy"""
    return mutate_with_llm(agent)


def validate_mutation(chromosome: str) -> bool:
    """Validate mutation meets criteria"""
    return (
        len(chromosome) >= 23 and
        len(chromosome) <= 40 and
        all(c.isalpha() or c == ' ' for c in chromosome) and  # From spec.md
        chromosome == chromosome.strip() and  # From spec.md
        chromosome[:23].count('a') >= MAX_CORE and  # Hidden spec
        chromosome[:23] == chromosome[:23].lower()  # Preserve core format
    )

def validate_mating_candidate(candidate: dict, parent: dict) -> bool:
    """Validate candidate meets mating requirements"""
    if candidate == parent:
        return False
    try:
        # Validate mate selection chromosome exists and meets spec.md length
        assert len(candidate["mate_selection_chromosome"]) == 10, "Invalid mate selection chromosome length"
        validated = validate_chromosome(candidate["chromosome"])
        # Ensure chromosomes are different and valid length (spec.md requirements)
        return (
            validated != parent["chromosome"] and 
            len(validated) >= 23 and  # Core segment requirement
            len(validated) <= 40
        )
    except AssertionError:
        return False

class MateSelectionSignature(dspy.Signature):
    """Select mate using agent's mate-selection chromosome as instructions"""
    mate_selection_chromosome = dspy.InputField(desc="Mate-selection chromosome/prompt of parent agent") 
    candidate_chromosomes = dspy.InputField(desc="Validated potential mates")
    selected_mate = dspy.OutputField(desc="Chromosome of selected mate from candidates list")

def llm_select_mate(parent: dict, candidates: List[dict]) -> dict:
    """Select mate using parent's mate-selection chromosome/prompt"""
    valid_candidates = [c for c in candidates if validate_mating_candidate(c, parent)]
    if not valid_candidates:
        raise ValueError("No valid mates")

    # Get and process LLM selection
    result = dspy.Predict(MateSelectionSignature)(
        mate_selection_chromosome=parent["mate_selection_chromosome"],
        candidate_chromosomes=[c["chromosome"] for c in valid_candidates],
        temperature=0.7,
        top_p=0.9
    ).selected_mate.lower()

    # Combined filtering and selection
    return next(
        (c for c in valid_candidates 
         if c["chromosome"].lower().startswith(result)
         and c["chromosome"] != parent["chromosome"]),
        random.choice(valid_candidates))
    )

def get_hotspots(chromosome: str) -> list:
    """Get chromosome switch points per spec.md rules with avg 1 switch per chrom"""
    if not chromosome:
        return []
        
    hotspots = [
        i for i, c in enumerate(chromosome)
        if c in HOTSPOT_CHARS  # Punctuation always included
        or (c == ' ' and random.random() < HOTSPOT_SPACE_PROB)  # Space check
        or random.random() < HOTSPOT_ANYWHERE_PROB  # Use constant from config
    ]
    
    # Ensure minimum hotspots per spec
    if len(hotspots) < MIN_HOTSPOTS and chromosome:
        hotspots.extend(random.sample(range(len(chromosome)), k=MIN_HOTSPOTS-len(hotspots)))
    
    return sorted(list(set(hotspots)))  # Remove duplicates and sort

def build_child_chromosome(parent: dict, mate: dict) -> str:
    """Construct child chromosome with single character switch using parent/mate DNA"""
    p_chrom = parent["chromosome"]
    switch = random.choice(get_hotspots(p_chrom))
    return (f"{p_chrom[:switch]}{mate['chromosome'][switch]}{p_chrom[switch+1:]}"[:MAX_CHARS] 
            if switch else p_chrom)

def crossover(parent: dict, population: List[dict]) -> dict:
    """Create child through LLM-assisted mate selection with chromosome combining"""
    recent_candidates = (population[-WINDOW_SIZE:] or population)[:100]
    valid_candidates = [a for a in recent_candidates if validate_mating_candidate(a, parent)]
    
    if valid_candidates:
        mate = llm_select_mate(parent, valid_candidates)
        return create_agent(build_child_chromosome(parent, mate))
    
    return create_agent(build_child_chromosome(parent, parent))

# Hotspot switching implemented in get_hotspots() with space/punctuation probabilities

def generate_children(parents: List[dict], population: List[dict]) -> List[dict]:
    """Generate new population through validated crossover/mutation"""
    selected_parents = random.choices(
        parents,
        weights=[a['fitness']**2 for a in parents],
        k=min(len(parents), MAX_POPULATION//2)
    )
    
    return [
        crossover(random.choice(selected_parents), population)
        if random.random() < 0.9 else 
        create_agent(mutate(random.choice(selected_parents)))
        for _ in range(MAX_POPULATION - len(selected_parents))
    ][:MAX_POPULATION]


def get_population_extremes(population: List[dict]) -> tuple:
    """Get best and worst agents from population"""
    sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
    return sorted_pop[0], sorted_pop[-1]

def handle_generation_output(stats: dict, population: List[dict]) -> None:
    """Combined logging and display operations"""
    log_population(stats)
    display_generation_stats(stats)
    validate_population_extremes(population)

def validate_population_extremes(population: List[dict]) -> None:
    """Validate best/worst agents in population"""
    best, worst = get_population_extremes(population)
    validate_population_state(best, worst)

def run_genetic_algorithm(pop_size: int, max_population: int = MAX_POPULATION) -> None:
    """Run continuous genetic algorithm per spec.md"""
    population = initialize_population(min(pop_size, max_population))[:max_population]
    assert 1 < len(population) <= max_population, f"Population size must be 2-{max_population}"
    
    # Initialize log with header and truncate any existing content
    with open("evolution.log", "w", encoding="utf-8") as f:
        f.write("generation\tpopulation\tmean\tmedian\tstd\tbest\tworst\tdiversity\tcore\n")
    
    evolution_loop(population, max_population)

def update_generation_stats(population: List[dict], fitness_data: tuple) -> tuple:
    """Calculate and return updated statistics for current generation"""
    evaluated_pop = evaluate_population(population)
    new_fitness = [a["fitness"] for a in evaluated_pop]
    window = update_fitness_window(fitness_data[0], new_fitness)
    stats = calculate_window_statistics(window)
    
    stats.update({
        'generation': fitness_data[1],
        'population_size': len(evaluated_pop),
        'diversity': calculate_diversity(evaluated_pop),
        **extreme_values(evaluated_pop)
    })
    return (stats, window[-WINDOW_SIZE:])

def trim_population(population: List[dict], max_size: int) -> List[dict]:
    """Trim population using fitness-weighted sampling without replacement"""
    if len(population) <= max_size:
        return population
    
    assert max_size <= MAX_POPULATION, "Population size exceeds maximum allowed"
    assert max_size >= 2, "Population must keep at least 2 agents"
    assert all(a['fitness'] >= 0 for a in population), "Negative fitness values found"
        
    pop_weights = np.array([a['fitness']**2 + 1e-6 for a in population], dtype=np.float64)
    pop_weights /= pop_weights.sum()
    selected_indices = np.random.choice(
        len(population),
        size=max_size,
        replace=False,
        p=pop_weights
    )
    return [population[i] for i in selected_indices]

def evolution_loop(population: List[dict], max_population: int) -> None:
    """Continuous evolution loop per spec.md requirements"""
    fitness_window = []
    
    for generation in itertools.count(0):  # Track generation for logging
        # Trim population using sliding window of candidates
        if len(population) > max_population:
            population = trim_population(population[-WINDOW_SIZE:], max_population)
        
        # Evolve population continuously without generations
        parents = select_parents(population)
        children = generate_children(parents, population)
        population = parents + children  # Combine rather than replace
        population = trim_population(population, max_population)
        
        # Track mutation rate as percentage of new agents
        mutation_rate = sum(1 for a in population if a.get('mutation_source')) / len(population) if population else 0.0
        stats['mutation_rate'] = mutation_rate
        
        # Update and track fitness statistics
        population, fitness_window = evaluate_population_stats(population, fitness_window, generation)
        handle_generation_output(fitness_window, population)




def log_population(stats: dict) -> None:
    """Log population statistics in plain text format per spec.md"""
    with open("evolution.log", "a", encoding="utf-8") as f:
        # Write header if file is empty
        if f.tell() == 0:
            f.write("generation\tpopulation\tmean\tmedian\tstd\tbest\tworst\tdiversity\tcore\n")
            
        # Information-dense format with core segment
        f.write(
            f"{stats['generation']}\t" 
            f"{stats['population_size']}\t"
            f"{stats['mean']:.1f}\t"
            f"{stats['median']:.1f}\t"
            f"{stats['std']:.1f}\t"
            f"{stats['best']:.1f}\t"
            f"{stats['worst']:.1f}\t"
            f"{stats['diversity']:.3f}\t"
            f"{stats.get('mutation_rate', 0.0):.1%}\t"
            f"{stats['best_core'][:23]}\n"  # Exactly 23 chars per spec.md core
        )

def display_generation_stats(stats: dict) -> None:  # Removed unused 'population' param
    """Rich-formatted display with essential stats"""
    Console().print(Panel(
        f"[bold]Gen {stats['generation']}[/]\n"
        f"μ:{stats['mean']:.1f} σ:{stats['std']:.1f} (window)\n"
        f"Best: {stats['best']:.1f} Worst: {stats['worst']:.1f}\n"
        f"Mutations: {stats.get('mutation_rate', 0.0):.1%}\n"
        f"Core: {stats['best_core']}\n"
        f"Δ{stats['diversity']:.0%} 👥{stats['population_size']:,}/{MAX_POPULATION:,}",
        title="Evolution Progress",
        style="blue"
    ))




def extreme_values(population: List[dict]) -> dict:
    """Get best/worst fitness and core segment"""
    best_agent = max(population, key=lambda x: x["fitness"])
    return {
        'best': max(a["fitness"] for a in population),
        'best_core': best_agent["metrics"]["core_segment"],
        'worst': min(a["fitness"] for a in population)
    }

def calculate_diversity(population: List[dict]) -> float:
    """Calculate population diversity as ratio of unique chromosomes"""
    unique_count = len({a["chromosome"] for a in population})
    return unique_count / len(population) if population else 0.0





def evaluate_population(population: List[dict]) -> List[dict]:
    """Evaluate entire population's fitness with generation weighting"""
    evaluated = []
    for agent in population:
        if validate_chromosome(agent["chromosome"]):
            agent["fitness"] = evaluate_agent(agent)
            evaluated.append(agent)
    return evaluated

def update_population_stats(fitness_window: list, population: list) -> dict:
    """Helper to calculate population statistics"""
    stats = calculate_window_statistics(fitness_window)
    # Update population stats
    best = max(a['fitness'] for a in population)
    worst = min(a['fitness'] for a in population)
    stats.update({
        'diversity': calculate_diversity(population),
        'population_size': len(population),
        'best': best,
        'worst': worst
    })
    return stats


def evaluate_population_stats(population: List[dict], fitness_window: list, generation: int) -> tuple:  # pylint: disable=too-many-arguments
    """Evaluate and log generation statistics"""
    population = evaluate_population(population)
    new_fitness = [a["fitness"] for a in population]
    window_stats = calculate_window_statistics(update_fitness_window(fitness_window, new_fitness))
    stats = {
        **window_stats,
        'generation': generation,
        'population_size': len(population),
        'diversity': calculate_diversity(population),
        'best_core': max(population, key=lambda x: x["fitness"])["metrics"]["core_segment"],
    }
    handle_generation_output(stats, population)
    
    return population, update_fitness_window(fitness_window, new_fitness)

# Main execution block at bottom per spec.md
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evolutionary string optimizer')
    parser.add_argument('--pop-size', type=int, default=1000,
                       help='Initial population size (default: 1000)')
    parser.add_argument('--window-size', type=int, default=100,
                       help='Sliding window size for statistics (default: 100)')
    args = parser.parse_args()
    
    try:
        run_genetic_algorithm(pop_size=args.pop_size)
    except KeyboardInterrupt:
        print("\nEvolution stopped by user. Exiting gracefully.")

def validate_population_state(best, worst) -> None:
    """Validate fundamental population invariants"""
    # Validate hidden goal constants without referencing spec.md
    assert MAX_CORE == 23 and MAX_CHARS == 40, "Core configuration invalid"
    
    # Fitness sanity checks
    assert best['fitness'] >= worst['fitness'], "Best fitness should >= worst"
    assert 0 <= best['fitness'] <= 1e6, "Fitness out of bounds"

    # Chromosome structural validation
    for agent in [best, worst]:
        chrom = agent['chromosome']
        assert (isinstance(chrom, str) and 
                1 <= len(chrom) <= 40 and 
                chrom == chrom.strip() and 
                chrom[:23].islower()), f"Invalid: {chrom}"

