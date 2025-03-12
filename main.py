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
    window = fitness_window[-WINDOW_SIZE:]  # Strictly last 100 evals
    
    if len(window) < WINDOW_SIZE:
        return {
            'mean': 0.0, 'median': 0.0, 'std': 0.0,
            'best_current': 0.0, 'worst_current': 0.0,
            'best_window': 0.0, 'worst_window': 0.0
        }

    arr = np.array(window, dtype=np.float64)
    return {
        'mean': float(np.nanmean(arr)),
        'median': float(np.nanmedian(arr)),
        'std': float(np.nanstd(arr)),
        'best_current': float(np.nanmax(arr)),
        'worst_current': float(np.nanmin(arr)),
        'best_window': float(np.nanmax(arr)),
        'worst_window': float(np.nanmin(arr))
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
    
    # Combined fitness^2 and Pareto weighting in single operation
    weights = (
        np.array([a['fitness']**2 for a in population], dtype=np.float64) * 
        (np.random.pareto(2.0, len(population)) + 1)  # Spec-compliant Pareto
    )
    
    # Numeric stability with vectorized operations
    weights = np.nan_to_num(weights, nan=1e-6).clip(1e-6, np.finfo(np.float64).max)  # pylint: disable=no-member
    total = weights.sum()
    assert not np.isclose(total, 0), "Weight total cannot be zero"
    return weights / total

def select_parents(population: List[dict]) -> List[dict]:
    """Select parents using Pareto distribution weighted by fitness^2 with weighted sampling without replacement"""
    if not population:
        return []
    
    # Enforce population limit first (spec.md requirement)
    population = population[:MAX_POPULATION]
    weights = calculate_parent_weights(population)
    
    # Convert weights to probabilities
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    
    # Weighted sampling without replacement using probability-ordered selection
    sample_size = min(len(population), MAX_POPULATION//2)
    sorted_indices = sorted(range(len(population)), key=lambda i: -probs[i])
    selected_indices = sorted_indices[:sample_size]
    
    return [population[i] for i in selected_indices]




class MutateSignature(dspy.Signature):
    """Mutate chromosomes while preserving first 23 characters and increasing 'a' density."""
    chromosome = dspy.InputField(desc="Current chromosome to mutate")
    instructions = dspy.InputField(desc="Mutation strategy instructions") 
    mutated_chromosome = dspy.OutputField(desc="Improved chromosome meeting requirements")

def mutate_with_llm(agent: dict) -> str:
    """Optimized LLM mutation with validation"""
    chromosome = agent["chromosome"]
    
    response = dspy.Predict(MutateSignature)(
        chromosome=agent["chromosome"],
        instructions=agent["mutation_chromosome"],
        temperature=min(1.0, float(agent["mutation_chromosome"][:3] or 0.7)),
        top_p=min(1.0, float(agent["mutation_chromosome"][3:7] or 0.9)),
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
        validate_chromosome(candidate["chromosome"])
        return True
    except AssertionError:
        return False

class MateSelectionSignature(dspy.Signature):
    """Select best mate candidate using agent's mating strategy chromosome."""
    parent_chromosome = dspy.InputField(desc="Mate-selection chromosome/prompt of parent agent")
    candidate_chromosomes = dspy.InputField(desc="Potential mates filtered by validation")
    selected_mate = dspy.OutputField(desc="Chromosome of selected mate from candidates list")

def llm_select_mate(parent: dict, candidates: List[dict]) -> dict:
    """Select mate using parent's mate-selection chromosome/prompt"""
    valid = [c for c in candidates if validate_mating_candidate(c, parent)]
    if not valid:
        raise ValueError("No valid mates")

    response = dspy.Predict(MateSelectionSignature)(
        parent_chromosome=parent["mate_selection_chromosome"],
        candidate_chromosomes=[c["chromosome"] for c in valid],
        temperature=0.7,
        top_p=0.9
    )

    return next(
        (c for c in valid if c["chromosome"] == response.selected_mate.strip()[:40]),
        random.choices(valid, weights=[c["fitness"]**2 + 1e-6 for c in valid])[0]
    )

def get_hotspots(chromosome: str) -> list:
    """Get chromosome switch points per spec.md rules (punctuation/space with 10% chance)"""
    # Force at least one hotspot and ensure ~1 switch per combination on average
    forced_hotspots = [i for i, c in enumerate(chromosome) if c in {'.', '!', '?', ' '}]
    random_hotspots = [i for i in range(len(chromosome)) if random.random() < 0.15]
    return list(set(forced_hotspots + random_hotspots)) or [0]

def get_hotspots(chromosome: str) -> list:
    """Get chromosome switch points per spec.md rules (punctuation/space with 10% chance)"""
    return [
        i for i, c in enumerate(chromosome)
        if c in {'.', '!', '?', ' '} or random.random() < 0.1
    ]

def build_child_chromosome(p_chrom: str, m_chrom: str, hotspots: list) -> str:
    """Construct child chromosome with single character switch"""
    switch_point = random.choice(hotspots) if hotspots else 0
    return f"{p_chrom[:switch_point]}{m_chrom[switch_point]}{p_chrom[switch_point+1:]}"[:40]

def crossover(parent: dict, population: List[dict]) -> dict:
    """Create child through LLM-assisted mate selection with chromosome switching"""
    candidates = (population[-WINDOW_SIZE:] or population)[:100]  # Hard limit
    valid_candidates = [a for a in candidates if validate_mating_candidate(a, parent)]
    
    # Fix syntax errors in random.choices call
    selected = random.choices(
        valid_candidates,
        weights=[a['fitness']**2 + 1e-6 for a in valid_candidates],
        k=min(5, len(valid_candidates))
    )
    
    mate = llm_select_mate(parent, selected)
    
    return create_agent(build_child_chromosome(
        parent["chromosome"],
        mate["chromosome"],
        get_hotspots(parent["chromosome"]) or [random.randint(0, len(parent["chromosome"])-1)])
    ))



def generate_children(parents: List[dict], population: List[dict]) -> List[dict]:
    """Generate new population through validated crossover/mutation"""
    next_gen = parents.copy()
    max_children = min(MAX_POPULATION - len(parents), len(parents)*2)
    
    # Enforce population limit before extending
    assert len(next_gen) <= MAX_POPULATION, "Population overflow before generation"
    
    next_gen.extend([
        (crossover(random.choice(parents), population)
        if random.random() < 0.9 else  # 90% crossover, 10% mutation
        create_agent(mutate(random.choice(parents))))
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
    with open("evolution.log", "w", encoding="utf-8") as f:
        pass  # Just truncate the file
    
    evolution_loop(population, max_population)

def update_generation_stats(population: List[dict], fitness_window: list, generation: int) -> tuple:
    """Calculate and return updated statistics for current generation"""
    evaluated_pop = evaluate_population(population)
    new_fitnesses = [a["fitness"] for a in evaluated_pop]
    updated_window = update_fitness_window(fitness_window, new_fitnesses)
    
    stats = calculate_window_statistics(updated_window)
    best_agent = max(evaluated_pop, key=lambda x: x["fitness"])
    
    stats.update({
        'generation': generation,
        'population_size': len(evaluated_pop),
        'diversity': calculate_diversity(evaluated_pop),
        'best': best_agent["fitness"],
        'best_core': best_agent["metrics"]["core_segment"],
        'worst': min(a["fitness"] for a in evaluated_pop)
    })
    return stats, updated_window

def evolution_loop(population: List[dict], max_population: int) -> None:
    """Continuous evolution loop with combined operations"""
    population = sorted(population, 
        key=lambda a: (-a["fitness"], -hash(a["chromosome"]))
    )[:max_population]
    
    fitness_window = []
    
    for generation in itertools.count(0):
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
    with open("evolution.log", "a", encoding="utf-8") as f:  # Using 'with' per pylint
        # Spec.md requires dense, minimal logging
        f.write(
            f"{stats['generation']}\t" 
            f"{stats['population_size']}\t"
            f"{stats['mean']:.1f}\t"
            f"{stats['median']:.1f}\t" 
            f"{stats['std']:.1f}\t"
            f"{stats['best_window']:.1f}\t"
            f"{stats['worst_window']:.1f}\n"
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

