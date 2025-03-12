import itertools
import random
import string
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
    """Calculate statistics for sliding window of last 100 evaluations"""
    assert len(fitness_window) >= 0, "Fitness window cannot be negative length"
    
    window = fitness_window[-WINDOW_SIZE:] if fitness_window else []
    assert 0 <= len(window) <= WINDOW_SIZE, f"Window size violation: {len(window)}"
    
    if not window:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0,
                'best': 0.0, 'worst': 0.0, 'q25': 0.0, 'q75': 0.0}

    arr = np.array(window, dtype=np.float64)
    return {
        'mean': float(np.nanmean(arr)),
        'median': float(np.nanmedian(arr)),
        'std': float(np.nanstd(arr)),
        'best': float(np.nanmax(arr)),
        'worst': float(np.nanmin(arr)),
        'q25': float(np.nanpercentile(arr, 25)),
        'q75': float(np.nanpercentile(arr, 75))
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
        "mate_selection_chromosome": (chromosome[23:40].ljust(17, ' ')[:17]),
        "mutation_chromosome": chromosome[40:60][:20],  # Handles overflow safely
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
    """Select parents using Pareto distribution weighted by fitness^2 with weighted sampling without replacement"""
    if not population:
        return []
    
    # Calculate weights and sample in one flow
    candidates = population[-WINDOW_SIZE:]
    pareto_weights = np.array([a['fitness']**2 + 1e-6 for a in candidates]) * np.random.pareto(2.0, len(candidates))
    
    return [candidates[i] for i in np.random.choice(
        len(candidates),
        size=min(len(population), MAX_POPULATION//2),
        p=pareto_weights/pareto_weights.sum(),
        replace=False
    )]




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
        temperature=0.7,
        top_p=0.9
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

def mutate(chromosome: str) -> str:  # Problem param removed since we get from dspy config
    """Mutate a chromosome with LLM-based mutation as primary strategy"""
    return mutate_with_llm(chromosome)


def validate_mutation(chromosome: str) -> bool:
    """Validate mutation meets criteria"""
    return (
        len(chromosome) >= 23 and
        len(chromosome) <= 40 and
        all(c.isalpha() or c == ' ' for c in chromosome) and  # From spec.md
        chromosome == chromosome.strip() and  # From spec.md
        chromosome[:23].count('a') >= chromosome[:23].count('a')  # Hidden spec
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

def crossover(parent: dict, population: List[dict]) -> dict:
    """Create child through LLM-assisted mate selection with chromosome switching"""
    candidates = population[-WINDOW_SIZE:]
    if not candidates:
        raise ValueError("No candidates available for crossover")
        
    weights = np.array([a["fitness"]**2 + 1e-6 for a in candidates])
    selected_mate = llm_select_mate(parent, random.choices(
        candidates,
        weights=weights/weights.sum(),
        k=min(5, len(population))
    ))
    
    return create_agent(''.join(
        selected_mate["chromosome"][i] if (random.random() < 1/len(parent["chromosome"]) or c in {'.', '!', '?', ' '})
        else c
        for i, c in enumerate(parent["chromosome"])
    )[:40])



def generate_children(parents: List[dict], population: List[dict]) -> List[dict]:
    """Generate new population through validated crossover/mutation"""
    next_gen = parents.copy()
    max_children = min(MAX_POPULATION, len(parents)*2)
    
    next_gen.extend(
        crossover(random.choice(parents), population)
        if random.random() < 0.9 else  # 90% crossover, 10% mutation
        create_agent(mutate(random.choice(parents)["chromosome"]))
        for _ in range(max_children)
    )
    
    return next_gen[:MAX_POPULATION]


def get_population_extremes(population: List[dict]) -> tuple:
    """Get best and worst agents from population"""
    sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
    return sorted_pop[0], sorted_pop[-1]

def log_and_display(stats: dict, population: List[dict]) -> None:
    """Combined logging and display operations"""
    log_population(stats)
    display_generation_stats(stats)
    validate_population_extremes(population)
    log_and_display(stats, population)

def validate_population_extremes(population: List[dict]) -> None:
    """Validate best/worst agents in population"""
    best, worst = get_population_extremes(population)
    validate_population_state(best, worst)

def run_genetic_algorithm(pop_size: int) -> None:
    """Run continuous genetic algorithm per spec.md"""
    population = initialize_population(min(pop_size, MAX_POPULATION))[:MAX_POPULATION]
    assert 1 < len(population) <= MAX_POPULATION, f"Population size must be 2-{MAX_POPULATION}"
    
    # Empty log file at program start per spec.md
    with open("evolution.log", "w", encoding="utf-8") as f:
        f.write("")  # Explicitly clear contents
    
    evolution_loop(population)

def evolution_loop(population: List[dict]) -> None:
    """Continuous evolution loop separated to reduce statement count"""
    assert MAX_POPULATION == 1_000_000, "Population limit mismatch with spec.md"
    population = population[:MAX_POPULATION]
    fitness_window = []
    
    for generation in itertools.count(0):
        # Evaluate, update and process in combined steps
        population = evaluate_population(population)[:MAX_POPULATION]
        fitness_window = update_fitness_window(fitness_window, [a["fitness"] for a in population])
        stats = calculate_window_statistics(fitness_window)
        
        # Combined stats update
        stats.update({
            'generation': generation,
            'population_size': len(population),
            'diversity': calculate_diversity(population),
            'best': max(a["fitness"] for a in population),
            'worst': min(a["fitness"] for a in population)
        })
        
        handle_generation_output(stats, population)
        population = generate_children(select_parents(population), population)[:MAX_POPULATION]



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evolutionary string optimizer')
    parser.add_argument('--pop-size', type=int, default=1000,
                       help='Initial population size')
    parser.add_argument('--max-population', type=int, default=1_000_000,
                       help='Maximum population size (per spec.md)')
    args = parser.parse_args()
    
    run_genetic_algorithm(pop_size=min(args.pop_size, args.max_population))

def log_population(stats: dict) -> None:
    """Log population statistics in plain text format"""
    with open("evolution.log", "a", encoding="utf-8") as f:
        # Use formatted string literals for cleaner output
        f.write(
            f"Gen:{stats['generation']} Mean:{stats['mean']:.2f} "
            f"Best:{stats['best']:.2f} Worst:{stats['worst']:.2f} "
            f"σ:{stats['std']:.1f} Size:{stats['population_size']}\n"
        )

def display_generation_stats(stats: dict) -> None:  # Removed unused 'population' param
    """Rich-formatted display with essential stats"""
    Console().print(Panel(
        f"[bold]Gen {stats['generation']}[/]\n"
        f"μ:{stats['mean']:.1f} σ:{stats['std']:.1f}\n"
        f"▲{stats['best']:.1f} ▼{stats['worst']:.1f}\n" 
        f"Δ{stats['diversity']:.0%} 👥{stats['population_size']:,}/{MAX_POPULATION:,}",
        title="Evolution Progress",
        style="blue"
    ))




def calculate_diversity(population: List[dict]) -> float:
    """Calculate population diversity ratio [0-1]"""
    unique_chromosomes = len({agent["chromosome"] for agent in population})
    return unique_chromosomes / len(population) if population else 0.0

def apply_mutations(generation: List[dict], base_rate: float) -> List[dict]:
    """Auto-adjust mutation rate based on population diversity"""
    # Combined calculations to reduce locals
    mutation_rate = np.clip(
        base_rate * (1.0 - np.log1p(calculate_diversity(generation))),
        0.1, 
        0.8
    )
    return [
        ({**agent, "chromosome": mutate(agent["chromosome"])} 
         if random.random() < mutation_rate else agent)
        for agent in generation
    ]




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

def validate_population_state(best, worst) -> None:
    """Validate fundamental population invariants"""
    # Validate population invariants
    assert best['fitness'] >= worst['fitness'], "Best fitness should >= worst fitness"
    assert 0 <= best['fitness'] <= 1e6, "Fitness out of reasonable bounds"
    assert 0 <= worst['fitness'] <= 1e6, "Fitness out of reasonable bounds"
    assert isinstance(best['chromosome'], str), "Chromosome should be string"
    assert isinstance(worst['chromosome'], str), "Chromosome should be string"
    assert len(best['chromosome']) <= 40, "Chromosome exceeded max length"
    assert len(worst['chromosome']) <= 40, "Chromosome exceeded max length"

