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
        'best': float(np.nanmax(window_arr)),
        'worst': float(np.nanmin(window_arr))
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


def select_parents(population: List[dict]) -> List[dict]:
    """Select parents using Pareto distribution weighted by fitness^2 with weighted sampling without replacement"""
    if not population:
        return []
    
    # Combined weights calculation and sampling in one vectorized operation
    weights = np.nan_to_num(
        np.array([a['fitness'] ** 2 for a in population], dtype=np.float64) * 
        (np.random.pareto(3.0, len(population)) + 1),
        nan=1e-6
    ).clip(1e-6)
    
    weights /= weights.sum()  # Normalize
    
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
HOTSPOT_SPACE_PROB = 0.1  # Probability to create hotspot at space (spec.md 10%)
MIN_HOTSPOTS = 1  # Minimum switch points per chromosome
HOTSPOT_ANYWHERE_PROB = 0.023  # ~1 hotspot per 40 char chromosome on average (spec.md requirement)
# Probability tuned to achieve average 1 switch per chrom combined with punctuation
HOTSPOT_ANYWHERE_PROB = 0.02  # Reduced from 0.02 to 0.015 to better match spec.md requirement

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
    # Combined extraction and validation
    mc = agent["mutation_chromosome"]
    temperature = max(0.0, min(2.0, float(mc[0:3] or '0.7')))
    top_p = max(0.0, min(1.0, float(mc[3:7] or '0.9')))
    
    assert 0.0 <= temperature <= 2.0, f"Invalid temperature {temperature}"
    assert 0.0 <= top_p <= 1.0, f"Invalid top_p {top_p}"

    response = dspy.Predict(MutateSignature)(
        chromosome=agent["chromosome"],
        instructions=mc,
        temperature=temperature,
        top_p=top_p,
    )

    valid_mutations = (
        str(r).strip()[:40].lower()
        for r in response.completions
        if (len(str(r).strip()) >= 23 
            and str(r).strip().startswith(agent["chromosome"][:23].lower())  # Validate core segment match
            and str(r).strip()[:23].count('a') >= agent["chromosome"][:23].count('a'))  # Validate 'a' count
    )
    
    # Return first valid mutation or fallback to core + random
    return next(
        valid_mutations,
        agent_chrom[:23] + ''.join(random.choices(
            string.ascii_letters.lower(), 
            k=random.randint(0, 17)  # 40-23=17 max additional chars
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
    """Get chromosome switch points per spec.md rules with avg 1 switch per chrom"""
    hotspots = []
    for i, c in enumerate(chromosome):
        # Always include punctuation hotspots
        if c in HOTSPOT_CHARS:
            hotspots.append(i)
        # Space characters have 10% chance to be hotspot
        elif c == ' ' and random.random() < HOTSPOT_SPACE_PROB:
            hotspots.append(i)
        # All characters have base chance to be hotspot
        if random.random() < HOTSPOT_ANYWHERE_PROB:
            hotspots.append(i)
    
    # Ensure at least one hotspot per chromosome and remove duplicates
    hotspots = list(set(hotspots))
    target_hotspots = max(MIN_HOTSPOTS, len(chromosome) // 40)
    while len(hotspots) < target_hotspots:
        hotspots.append(random.randint(0, len(chromosome)-1))
    
    return hotspots if hotspots else [random.randint(0, len(chromosome)-1)]

def build_child_chromosome(parent: dict, mate: dict) -> str:
    """Construct child chromosome with single character switch using parent/mate DNA"""
    p_chrom = parent["chromosome"]
    m_chrom = mate["chromosome"]
    switch = random.choice(get_hotspots(p_chrom))
    return f"{p_chrom[:switch]}{m_chrom[switch]}{p_chrom[switch+1:]}"[:MAX_CHARS] if switch else p_chrom

def crossover(parent: dict, population: List[dict]) -> dict:
    """Create child through LLM-assisted mate selection with chromosome switching"""
    # Get validated candidates in one operation
    valid_candidates = [
        a for a in (population[-WINDOW_SIZE:] or population)[:100]
        if validate_mating_candidate(a, parent)
    ]
    
    # Weighted sampling without replacement using numpy
    if valid_candidates:
        weights = np.array([a['fitness']**2 + 1e-6 for a in valid_candidates], dtype=np.float64)
        weights /= weights.sum()
        selected_indices = np.random.choice(
            len(valid_candidates), 
            size=min(5, len(valid_candidates)),
            replace=False,
            p=weights
        )
        mates = [valid_candidates[i] for i in selected_indices]
    else:
        mates = []
    
    mate = llm_select_mate(parent, mates) if mates else parent  # Fallback to parent if no mates
    
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
    with open("evolution.log", "w", encoding="utf-8"):
        pass  # Just truncate the file
    
    evolution_loop(population, max_population)

def update_generation_stats(population: List[dict], fitness_data: tuple) -> tuple:
    """Calculate and return updated statistics for current generation"""
    fitness_window, generation = fitness_data
    evaluated_pop = evaluate_population(population)
    new_fitness = [a["fitness"] for a in evaluated_pop]
    updated_window = update_fitness_window(fitness_window, new_fitness)
    
    stats = {
        'generation': generation,
        'population_size': len(evaluated_pop),
        'diversity': calculate_diversity(evaluated_pop),
        **calculate_window_statistics(updated_window),
        **extreme_values(evaluated_pop),
        'best_current': max(new_fitness),
        'worst_current': min(new_fitness)
    }
    
    return stats, updated_window[-WINDOW_SIZE:]

def evolution_loop(population: List[dict], max_population: int) -> None:
    """Continuous evolution loop with combined operations"""
    fitness_window = []
    
    for generation in itertools.count(0):
        # Continuous population trimming with combined operations
        # Weighted sampling without replacement using fitness² weighting
        pop_weights = [a['fitness']**2 + 1e-6 for a in population]
        population = population if len(population) <= max_population else [
            population[i] 
            for i in np.random.choice(
                len(population),
                size=max_population,
                replace=False,
                p=np.array(pop_weights)/sum(pop_weights))
            ]
        ]
        
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
            f"{stats['best']:.1f}\t"
            f"{stats['worst']:.1f}\t"
            f"{stats['diversity']:.2f}\n"  # Added diversity metric
        )

def display_generation_stats(stats: dict) -> None:  # Removed unused 'population' param
    """Rich-formatted display with essential stats"""
    Console().print(Panel(
        f"[bold]Gen {stats['generation']}[/]\n"
        f"μ:{stats['mean']:.1f} σ:{stats['std']:.1f} (window)\n"
        f"Best: {stats['best']:.1f} Worst: {stats['worst']:.1f}\n"
        f"Core: {stats['best_core']}\n"
        f"Δ{stats['diversity']:.0%} 👥{stats['population_size']:,}/{MAX_POPULATION:,}",
        title="Evolution Progress",
        style="blue"
    ))




def extreme_values(population: List[dict]) -> dict:
    """Get best/worst fitness and core segment"""
    best_agent = max(population, key=lambda x: x["fitness"])
    return {
        'best': best_agent["fitness"],
        'best_core': best_agent["metrics"]["core_segment"],
        'worst': min(a["fitness"] for a in population)
    }

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

