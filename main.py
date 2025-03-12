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
DEBUG_MODE = False  # Control debug output
WINDOW_SIZE = 100  # Sliding window size from spec.md
lm = dspy.LM(
    "openrouter/google/gemini-2.0-flash-001", max_tokens=40, timeout=10, cache=False
)
dspy.configure(lm=lm)

# Validate configuration
assert isinstance(lm, dspy.LM), "LM configuration failed"


def calculate_window_statistics(fitness_window: list) -> dict:
    """Calculate statistics for sliding window of last 100 evaluations (spec.md requirement)"""
    window_arr = np.array(fitness_window[-WINDOW_SIZE:], dtype=np.float64)
    return {
        'mean': float(np.nanmean(window_arr)),
        'median': float(np.nanmedian(window_arr)),
        'std': float(np.nanstd(window_arr)),
        'best_current': float(np.nanmax(window_arr)),
        'worst_current': float(np.nanmin(window_arr)),
        'best_window': float(np.nanmax(window_arr)),
        'worst_window': float(np.nanmin(window_arr))
    }

def update_fitness_window(fitness_window: list, new_fitnesses: list) -> list:
    """Maintain sliding window of last 100 evaluations"""
    return (fitness_window + new_fitnesses)[-100:]  # Fixed window size from spec

def score_chromosome(chromosome: str) -> dict:
    """Calculate structural scoring metrics"""
    core = chromosome[:23].lower()
    assert len(core) == 23, "Core segment must be 23 characters"
    
    # Calculate a_count and repeating pairs using optimized methods
    a_count = core.count('a')
    repeats = sum(core[i] == core[i-1] for i in range(1, len(core)))
    
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
    assert all(c in string.ascii_letters + " " for c in chromosome), "Invalid characters"
    
    # Generate random chromosome if empty
    if not chromosome:
        chromosome = "".join(random.choices(string.ascii_letters + " ", 
                                k=random.randint(20, 40)))
    
    # Split into three specialized chromosomes per spec.md
    return {
        "chromosome": chromosome,
        "task_chromosome": chromosome[:23],
        "mate_selection_chromosome": chromosome[23:33].ljust(10, ' ')[:10],  # 10 chars for mate selection per spec.md
        "mutation_chromosome": chromosome[33:40].ljust(7, ' ')[:7],  # 7 chars for mutation instructions
        "fitness": 0.0
    }


def evaluate_agent(agent: dict) -> float:
    """Evaluate agent fitness based on hidden optimization target"""
    chromo = validate_chromosome(agent["chromosome"])
    metrics = score_chromosome(chromo)
    
    # Combined fitness calculation to reduce locals
    agent["fitness"] = metrics['a_density'] * 46 - len(chromo) - 23
    
    assert len(metrics['core_segment']) == 23, "Core segment length mismatch"
    agent["metrics"] = metrics
    return agent["fitness"]


def initialize_population(pop_size: int) -> List[dict]:
    """Create initial population with random chromosomes using vectorized operations"""
    # Generate lengths first 
    lengths = [random.randint(20, 40) for _ in range(pop_size)]
    # Batch create all chromosomes
    chromosomes = [
        "".join(random.choices(string.ascii_letters + " ", k=length))
        for length in lengths
    ]
    # Parallel create agents
    return [create_agent(c) for c in chromosomes]


def calculate_parent_weights(population: List[dict]) -> np.ndarray:
    """Calculate parent selection weights with Pareto distribution and fitness^2"""
    assert len(population) > 0, "Cannot calculate weights for empty population"
    
    # Combined fitness^2 and Pareto weighting with validation
    fitness_scores = np.array([a['fitness'] ** 2 for a in population], dtype=np.float64)
    assert not np.isnan(fitness_scores).any(), "NaN values detected in fitness scores"
    
    # Pareto distribution with shape parameter=3.0 as specified (fitness^2 * Pareto)
    weights = fitness_scores * (np.random.pareto(3.0, len(population)) + 1)
    
    # Numeric stability with vectorized operations
    weights = np.nan_to_num(weights, nan=1e-6).clip(1e-6, np.finfo(np.float64).max, axis=0)
    total = weights.sum()
    assert total > 0 or len(population) == 0, "Weight total should be positive for non-empty population"
    return weights / total if total > 0 else np.ones_like(weights)/len(weights)

def select_parents(population: List[dict]) -> List[dict]:
    """Select parents using Pareto distribution weighted by fitness^2 with weighted sampling without replacement"""
    if not population:
        return []
    
    # Optimized sampling with numpy and weight validation
    population = population[:MAX_POPULATION]
    return [
        population[i] for i in np.random.choice(
            len(population),
            size=min(len(population), MAX_POPULATION//2),
            replace=False,
            p=calculate_parent_weights(population)
        )
    ]

# Configuration constants from spec.md
MUTATION_RATE = 0.1  # Base mutation probability 
HOTSPOT_CHARS = {'.', '!', '?', ' '}
HOTSPOT_SPACE_PROB = 0.1  # Probability to create hotspot at space (spec.md 10%)
MIN_HOTSPOTS = 1  # Minimum switch points per chromosome

def validate_mutation_rate(chromosome: str) -> None:
    """Ensure mutation parameters stay within valid ranges"""
    temp = float(chromosome[:3] or 0.7)
    top_p = float(chromosome[3:7] or 0.9)
    assert 0.0 <= temp <= 2.0, f"Invalid temperature {temp}"
    assert 0.0 <= top_p <= 1.0, f"Invalid top_p {top_p}"




class MutateSignature(dspy.Signature):
    """Mutate chromosomes while preserving first 23 characters and increasing 'a' density."""
    chromosome = dspy.InputField(desc="Current chromosome to mutate")
    instructions = dspy.InputField(desc="Mutation strategy instructions") 
    mutated_chromosome = dspy.OutputField(desc="Improved chromosome meeting requirements")

def mutate_with_llm(agent: dict) -> str:
    """Optimized LLM mutation with validation"""
    chromosome = agent["chromosome"]
    
    # Validate mutation parameters from chromosome
    raw_temp = float(agent["mutation_chromosome"][:3] or 0.7)
    raw_top_p = float(agent["mutation_chromosome"][3:7] or 0.9)
    temperature = max(0.0, min(1.0, raw_temp))
    top_p = max(0.0, min(1.0, raw_top_p))
    
    response = dspy.Predict(MutateSignature)(
        chromosome=agent["chromosome"],
        instructions=agent["mutation_chromosome"],
        temperature=temperature,
        top_p=top_p,
    )
    
    # Validate mutations with generator expression
    valid_mutations = (
        str(r).strip()[:40].lower()
        for r in response.completions
        if (len(str(r).strip()) >= 23 
            and str(r).strip().startswith(chromosome[:23].lower())
            and str(r).strip()[:23].count('a') >= chromosome[:23].count('a'))
    )

    # Return first valid mutation or fallback
    return next(valid_mutations, 
        chromosome[:23] + ''.join(random.choices(
            string.ascii_letters.lower(), 
            k=max(0, len(chromosome)-23)
        ))
    )

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
    """Select mate using DNA-loaded candidates and mate-selection chromosome (spec.md mating)"""
    parent_chromosome = dspy.InputField(desc="Mate-selection chromosome/prompt of parent agent")
    candidate_chromosomes = dspy.InputField(desc="Potential mates filtered by validation")
    selected_mate = dspy.OutputField(desc="Chromosome of selected mate from candidates list")

def llm_select_mate(parent: dict, candidates: List[dict]) -> dict:
    """Select mate using parent's mate-selection chromosome/prompt"""
    valid = [c for c in candidates if validate_mating_candidate(c, parent)]
    if not valid:
        raise ValueError("No valid mates")

    # Get LLM selection
    result = dspy.Predict(MateSelectionSignature)(
        parent_chromosome=parent["mate_selection_chromosome"],
        candidate_chromosomes=[c["chromosome"] for c in valid],
        temperature=0.7,
        top_p=0.9
    ).selected_mate

    # Find best match from LLM response
    return max(valid, key=lambda x: x["chromosome"].lower().startswith(result.lower()))

def get_hotspots(chromosome: str) -> list:
    """Get chromosome switch points per spec.md rules (punctuation/space with 10% chance)"""
    hotspots = []
    for i, c in enumerate(chromosome):
        # Always include punctuation hotspots
        if c in HOTSPOT_CHARS:
            hotspots.append(i)
        # Include spaces with 10% probability
        elif c == ' ' and random.random() < HOTSPOT_SPACE_PROB:
            hotspots.append(i)
        # Small chance to create hotspot anywhere
        elif random.random() < 0.02:  # ~1 hotspot per 50 chars on average
            hotspots.append(i)
    
    # Ensure minimum hotspots and return
    return hotspots or [random.randint(0, len(chromosome)-1)]

def build_child_chromosome(parent: dict, mate: dict) -> str:
    """Construct child chromosome with multiple switches using hotspots (spec.md average 1 per chrom)"""
    p_chrom = parent["chromosome"]
    m_chrom = mate["chromosome"]
    hotspots = get_hotspots(p_chrom)
    
    # Create switches at ~1 hotspot per chromosome on average
    chrom_parts = []
    last_pos = 0
    for pos in sorted(random.sample(hotspots, min(len(hotspots), len(p_chrom)//40 + 1))):
        chrom_parts.append(m_chrom[last_pos:pos+1] if random.random() < 0.5 else p_chrom[last_pos:pos+1])
        last_pos = pos + 1
    
    chrom_parts.append(m_chrom[last_pos:] if random.random() < 0.5 else p_chrom[last_pos:])
    return ''.join(chrom_parts)[:MAX_CHARS]

def crossover(parent: dict, population: List[dict]) -> dict:
    """Create child through LLM-assisted mate selection with chromosome switching"""
    # Get validated candidates in one operation
    valid_candidates = [
        a for a in (population[-WINDOW_SIZE:] or population)[:100]
        if validate_mating_candidate(a, parent)
    ]
    
    # Select mate and build child in streamlined way
    mate = llm_select_mate(parent, random.choices(
        valid_candidates,
        weights=[a['fitness']**2 + 1e-6 for a in valid_candidates],
        k=min(5, len(valid_candidates))
    ))
    
    return create_agent(build_child_chromosome(parent, mate))

# Hotspot switching implemented in get_hotspots() with space/punctuation probabilities

def generate_children(parents: List[dict], population: List[dict]) -> List[dict]:
    """Generate new population through validated crossover/mutation"""
    # Trim population using weighted sampling to maintain diversity
    parents = random.choices(
        parents,
        weights=[a['fitness']**2 for a in parents],
        k=min(len(parents), MAX_POPULATION//2)
    )
    next_gen = []
    max_children = MAX_POPULATION - len(parents)
    
    # Enforce population limit before generation (spec.md requirement)
    assert len(parents) <= MAX_POPULATION//2, "Parent population exceeds limit"
    
    # Enforce population limit before extending
    assert len(next_gen) <= MAX_POPULATION, "Population overflow before generation"
    
    next_gen.extend([
        crossover(random.choice(parents), population)
        if random.random() < 0.9 else  # 90% crossover, 10% mutation
        create_agent(mutate(random.choice(parents)))
        for _ in range(max_children)
    ])
    
    # Hard limit enforcement per spec.md
    next_gen = next_gen[:MAX_POPULATION]
    assert len(next_gen) <= MAX_POPULATION, "Population exceeded MAX_POPULATION after generation"
    
    return next_gen[:MAX_POPULATION]


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
    
    # Empty log file using with statement
    with open("evolution.log", "w", encoding="utf-8") as _:  # File is intentionally empty
        pass  # Truncate file without keeping reference
    
    evolution_loop(population, max_population)

def update_generation_stats(population: List[dict], fitness_data: tuple) -> tuple:
    """Calculate and return updated statistics for current generation"""
    fitness_window, generation = fitness_data
    evaluated_pop = evaluate_population(population)
    
    # Get all fitness scores first
    fitness_scores = [a["fitness"] for a in evaluated_pop]
    
    # Combine stats calculation
    return {
        **calculate_window_statistics(
            update_fitness_window(fitness_window, fitness_scores)),
        'generation': generation,
        'population_size': len(evaluated_pop),
        'diversity': calculate_diversity(evaluated_pop),
        'best': max(fitness_scores),
        'best_core': max(evaluated_pop, key=lambda x: x["fitness"])["metrics"]["core_segment"],
        'worst': min(fitness_scores)
    }, fitness_window[-WINDOW_SIZE:]

def evolution_loop(population: List[dict], max_population: int) -> None:
    """Continuous evolution loop with combined operations"""
    fitness_window = []
    
    for generation in itertools.count(0):
        # Trim population before each iteration (spec.md population limit)
        population = sorted(population, key=lambda x: x["fitness"], reverse=True)[:max_population]
        population = evaluate_population(population)
        fitness_window = update_fitness_window(fitness_window, [a["fitness"] for a in population])
        
        handle_generation_output({
            **calculate_window_statistics(fitness_window),
            'generation': generation,
            'population_size': len(population),
            'diversity': calculate_diversity(population),
            'best': max(population, key=lambda x: x["fitness"])["fitness"],
            'best_core': max(population, key=lambda x: x["fitness"])["metrics"]["core_segment"],
            'worst': min(a["fitness"] for a in population)
        }, population)
        
        population = generate_children(select_parents(population), population)[:MAX_POPULATION]




def log_population(stats: dict) -> None:
    """Log population statistics in plain text format per spec.md"""
    with open("evolution.log", "a", encoding="utf-8") as f:
        # Minimal format: gen pop_size mean median std best worst
        f.write(
            f"{stats['generation']}\t" 
            f"{stats['population_size']}\t"
            f"{stats['mean']:.1f}\t"
            f"{stats['median']:.1f}\t"
            f"{stats['std']:.1f}\t"
            f"{stats['best_window']:.1f}\t"
            f"{stats['worst_window']:.1f}\t"
            f"{stats['diversity']:.2f}\n"  # Added diversity metric
        )

def display_generation_stats(stats: dict) -> None:  # Removed unused 'population' param
    """Rich-formatted display with essential stats"""
    Console().print(Panel(
        f"[bold]Gen {stats['generation']}[/]\n"
        f"μ:{stats['mean']:.1f} σ:{stats['std']:.1f} (window)\n"
        f"▲{stats['best_window']:.1f} ▼{stats['worst_window']:.1f}\n"
        f"▲{stats['best_current']:.1f} ▼{stats['worst_current']:.1f}\n" 
        f"Δ{stats['diversity']:.0%} 👥{stats['population_size']:,}/{MAX_POPULATION:,}",
        title="Evolution Progress",
        style="blue"
    ))




def calculate_diversity(population: List[dict]) -> float:
    """Calculate population diversity ratio [0-1]"""
    unique_chromosomes = len({agent["chromosome"] for agent in population})
    return unique_chromosomes / len(population) if population else 0.0





def evaluate_population(population: List[dict]) -> List[dict]:
    """Evaluate entire population's fitness with generation weighting"""
    for agent in population:
        evaluate_agent(agent)
    return population

def update_population_stats(fitness_window: list, population: list) -> dict:
    """Helper to calculate population statistics"""
    stats = calculate_window_statistics(fitness_window)
    # Combined stats updates
    stats.update({
        'diversity': calculate_diversity(population),
        'population_size': len(population),
        'best': max(a['fitness'] for a in population),
        'worst': min(a['fitness'] for a in population)
    })
    return stats

# Main execution block at end per Python best practices
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evolutionary string optimizer')
    parser.add_argument('--pop-size', type=int, default=1000,
                       help='Initial population size')
    parser.add_argument('--max-population', type=int, default=1_000_000,
                       help='Maximum population size (per spec.md)')
    args = parser.parse_args()
    
    try:
        run_genetic_algorithm(pop_size=min(args.pop_size, args.max_population))
    except KeyboardInterrupt:
        print("\nEvolution stopped by user. Exiting gracefully.")

def validate_population_state(best, worst) -> None:
    """Validate fundamental population invariants per spec.md"""
    # Validate constants in single assertion
    assert all([
        MAX_CORE == 23,
        MAX_CHARS == 40,
        MAX_POPULATION == 1_000_000
    ]), "Critical constants modified"
    
    # Validate fitness relationships
    assert best['fitness'] >= worst['fitness'], "Best fitness should >= worst fitness"
    assert 0 <= best['fitness'] <= 1e6 and 0 <= worst['fitness'] <= 1e6, "Fitness out of bounds"
    
    # Combined chromosome validation
    for agent in [best, worst]:
        chrom = agent['chromosome']
        assert (
            isinstance(chrom, str) and
            1 <= len(chrom) <= 40 and
            chrom == chrom.strip() and
            all(c.isalpha() or c == ' ' for c in chrom) and
            chrom[:23].islower()
        ), f"Invalid chromosome in {agent}"
    
    # Validate core segment metrics
    assert (
        len(best['metrics']['core_segment']) == 23 and
        ' ' not in best['chromosome'].strip()
    ), "Core segment validation failed"
    
    # Validate mutation rate parameters are within sane bounds
    assert 0 <= best['fitness'] <= 1e6, "Fitness out of reasonable bounds"

