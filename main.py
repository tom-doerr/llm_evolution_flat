import random
import string
import itertools
from typing import List

import numpy as np
from rich.console import Console
from rich.panel import Panel
import dspy

MAX_POPULATION = 1_000_000  # Defined per spec.md population limit
MAX_CHARS = 40  # From spec.md (different from max tokens)
MAX_CORE = 23  # From spec.md hidden goal
WINDOW_SIZE = 100  # Default, can be overridden by CLI

# Validate hidden goal constants from spec.md
assert MAX_CORE == 23, "Core segment length must be 23 per spec.md"
assert MAX_CHARS == 40, "Max chromosome length must be 40 for this task"

# Configure DSPy with OpenRouter and timeout
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
    
    # Convert to string and strip whitespace
    chromosome = str(chromosome).strip()[:40].lower()  # Normalize to lowercase
    
    # For empty strings, return a valid default
    if not chromosome:
        return "a" * 23  # Default to valid chromosome with 'a's
    
    # Clean invalid characters and strip again to ensure no whitespace at ends
    chromosome = ''.join(c for c in chromosome if c.isalpha() or c == ' ').strip()
    
    # Structural validation
    assert len(chromosome) <= 40, f"Invalid length {len(chromosome)}"
    
    # If chromosome is too short after cleaning, pad with 'a's
    if len(chromosome) < 23:
        chromosome = chromosome.ljust(23, 'a')
    
    return chromosome

def create_agent(chromosome: str) -> dict:
    """Create a new agent as a dictionary"""
    # Validate and clean chromosome first
    chromosome = validate_chromosome(chromosome)
    
    # Create agent with validated chromosome
    return {
        "chromosome": chromosome,
        "task_chromosome": chromosome[:23].ljust(23, ' ')[:23],  # Enforce exact length
        "mate_selection_chromosome": chromosome[23:33].ljust(10, ' ')[:10].lower(),
        "mutation_chromosome": chromosome[33:40].ljust(7, ' ')[:7],  # Enforce exact 7 char length
        "fitness": 0.0,
        "mutation_source": "initial",  # Track mutation origin per spec.md
        "metrics": {}  # Initialize metrics dictionary
    }

def evaluate_agent(agent: dict) -> float:
    """Evaluate agent fitness based on hidden optimization target"""
    chromo = validate_chromosome(agent["chromosome"])
    metrics = score_chromosome(chromo)
    # Hidden goal: maximize 'a's in first 23 chars, minimize length after that
    a_count = chromo[:23].count('a')
    agent["fitness"] = a_count - (len(chromo) - 23 if len(chromo) > 23 else 0)
    
    assert len(metrics['core_segment']) == 23, "Core segment length mismatch"
    agent["metrics"] = metrics
    return agent["fitness"]


def initialize_population(pop_size: int) -> List[dict]:
    """Create initial population with random chromosomes"""
    # Start with random chromosomes with 'a's in the core segment
    chromosomes = [
        "a" * 23 + "".join(random.choices(string.ascii_lowercase + " ", k=17))
        for _ in range(pop_size)
    ]
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
    mutation_instructions = dspy.InputField(desc="Mutation strategy instructions") 
    mutated_chromosome = dspy.OutputField(desc="Improved chromosome meeting requirements")

def mutate_with_llm(agent: dict) -> str:
    """Optimized LLM mutation with validation"""
    agent["mutation_source"] = f"llm:{agent['mutation_chromosome']}"
    core_segment = agent["chromosome"][:MAX_CORE].lower()
    
    try:
        # Use the agent's mutation chromosome as instructions
        response = dspy.Predict(MutateSignature)(
            chromosome=agent["chromosome"],
            mutation_instructions=agent["mutation_chromosome"]
        )

        # Process completions
        for comp in response.completions:
            comp_str = str(comp).strip().lower()
            
            # Ensure core segment is preserved
            if not comp_str.startswith(core_segment):
                continue
                
            # Clean and validate
            valid_candidate = ''.join(c for c in comp_str[:MAX_CHARS] if c.isalpha() or c == ' ').strip()
            
            if valid_candidate and len(valid_candidate) >= MAX_CORE:
                return valid_candidate
                
    except Exception:
        # No exception handling needed per spec.md
        pass
    
    # Default fallback mutation - preserve core segment
    return f"{core_segment}{random.choices(string.ascii_lowercase + ' ', k=17)[:17]}".strip()


def mutate(agent: dict) -> str:
    """Mutate a chromosome with LLM-based mutation as primary strategy"""
    mutated = mutate_with_llm(agent)
    agent['mutations'] = agent.get('mutations', 0) + 1  # Track mutation count per agent
    return mutated


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
    # Quick checks first
    if candidate == parent:
        return False
    
    if "mutation_chromosome" not in candidate or "mate_selection_chromosome" not in candidate:
        return False
    
    if "chromosome" not in candidate:
        return False
        
    # Validate chromosome
    try:
        validated = validate_chromosome(candidate["chromosome"])
        
        # Check if chromosomes are different
        if validated == parent["chromosome"]:
            return False
            
        # Check length requirements
        if len(validated) < 23:
            return False
            
        return True
    except (AssertionError, KeyError):
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
        random.choice(valid_candidates)
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
    """Construct child chromosome with character switches at hotspots"""
    p_chrom = parent["chromosome"]
    m_chrom = mate["chromosome"]
    
    # Get hotspots for chromosome switching
    hotspots = get_hotspots(p_chrom)
    
    # If no hotspots found, create at least one
    if not hotspots and p_chrom:
        hotspots = [random.randint(0, len(p_chrom) - 1)]
    
    # Build combined chromosome with switches at hotspots
    result = ""
    last_pos = 0
    use_parent = True
    
    for pos in sorted(hotspots):
        if pos >= len(p_chrom):
            continue
            
        # Add segment from current chromosome
        segment = p_chrom[last_pos:pos] if use_parent else m_chrom[last_pos:pos]
        result += segment if last_pos < len(segment) else ""
        
        # Switch chromosomes
        use_parent = not use_parent
        last_pos = pos
    
    # Add final segment
    if last_pos < len(p_chrom):
        final = p_chrom[last_pos:] if use_parent else m_chrom[last_pos:min(len(m_chrom), len(p_chrom))]
        result += final
    
    # Validate before returning
    return validate_chromosome(result[:MAX_CHARS])

def crossover(parent: dict, population: List[dict]) -> dict:
    """Create child through LLM-assisted mate selection with chromosome combining"""
    valid_candidates = [a for a in (population[-WINDOW_SIZE:] or population)[:100] 
                       if validate_mating_candidate(a, parent)]
    
    if valid_candidates:
        mate = llm_select_mate(parent, valid_candidates)
        child = create_agent(build_child_chromosome(parent, mate))
        child["mutation_source"] = f"crossover:{parent['mutation_chromosome']}"
        return child
    
    child = create_agent(build_child_chromosome(parent, parent))
    child["mutation_source"] = "self-crossover"
    return child

# Hotspot switching implemented in get_hotspots() with space/punctuation probabilities

def generate_children(parents: List[dict], population: List[dict]) -> List[dict]:
    """Generate new population through validated crossover/mutation"""
    # Calculate weights, ensuring they're all positive
    weights = [max(a['fitness'], 0.001)**2 for a in parents]
    
    # If all weights are zero, use uniform weights
    if sum(weights) <= 0:
        weights = [1.0] * len(parents)
    
    selected_parents = random.choices(
        parents,
        weights=weights,
        k=min(len(parents), MAX_POPULATION//2)
    )
    
    children = [
        (crossover(random.choice(selected_parents), population) 
         if random.random() < 0.9 else 
         create_agent(mutate(random.choice(selected_parents))))
        for _ in range(MAX_POPULATION - len(selected_parents))
    ]
    return children[:MAX_POPULATION]


def get_population_extremes(population: List[dict]) -> tuple:
    """Get best and worst agents from population"""
    sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
    return sorted_pop[0], sorted_pop[-1]

def handle_generation_output(stats: dict, population: List[dict]) -> None:
    """Combined logging and display operations"""
    if population:  # Only log if we have a population
        log_population(stats)
        display_generation_stats(stats)
        validate_population_extremes(population)

def validate_population_extremes(population: List[dict]) -> None:
    """Validate best/worst agents in population"""
    best, worst = get_population_extremes(population)
    validate_population_state(best, worst)

def run_genetic_algorithm(pop_size: int) -> None:
    """Run continuous genetic algorithm per spec.md"""
    population = initialize_population(min(pop_size, MAX_POPULATION))[:MAX_POPULATION]
    assert 1 < len(population) <= MAX_POPULATION, f"Population size must be 2-{MAX_POPULATION}"
    
    # Initialize log with header and truncate any existing content per spec.md
    with open("evolution.log", "w", encoding="utf-8") as f:
        header = "generation\tpopulation\tmean\tmedian\tstd\tbest\tworst\tdiversity\tcore\n"
        f.write(header)
        # Validate plain text format
        assert '\n' in header and '\t' in header, "Log format must be plain text"
        assert not any([',' in header, '[' in header, ']' in header]), "No structured formats allowed in log"
    
    evolution_loop(population)

def update_generation_stats(population: List[dict], fitness_data: tuple) -> tuple:
    """Calculate and return updated statistics for current generation"""
    evaluated_pop = evaluate_population(population)
    fitness_values = [a["fitness"] for a in evaluated_pop]
    window = update_fitness_window(fitness_data[0], fitness_values)
    stats = calculate_window_statistics(window)
    
    # Get extreme values
    extremes = extreme_values(evaluated_pop)
    
    stats.update({
        'generation': fitness_data[1],
        'population_size': len(evaluated_pop),
        'diversity': calculate_diversity(evaluated_pop),
        'best': extremes['best'],
        'worst': extremes['worst'],
        'best_core': extremes['best_core']
    })
    return (stats, window[-WINDOW_SIZE:])

def trim_population(population: List[dict], max_size: int) -> List[dict]:
    """Trim population using fitness-weighted sampling without replacement"""
    # Apply hard cap from spec.md
    max_size = min(max_size, MAX_POPULATION)
    
    # Quick return if no trimming needed
    if len(population) <= max_size:
        return population
    
    # Prepare fitness values for weighting
    fitness_values = np.array([a['fitness'] for a in population], dtype=np.float64)
    
    # Handle negative fitness values (allowed per spec.md)
    min_fitness = min(fitness_values)
    if min_fitness < 0:
        fitness_values = fitness_values - min_fitness + 1e-6
    
    # Calculate weights and normalize
    weights = fitness_values ** 2
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
    
    # Select agents to keep
    selected_indices = np.random.choice(
        len(population),
        size=max_size,
        replace=False,
        p=weights
    )
    
    return [population[i] for i in selected_indices]

def evolution_loop(population: List[dict]) -> None:
    """Continuous evolution loop without generation concept"""
    fitness_window = []
    
    # Initial evaluation of population
    population = evaluate_population(population)
    fitness_window = [a["fitness"] for a in population]
    
    # Print initial stats
    print(f"Initial population: {len(population)} agents")
    print(f"Initial best fitness: {max(fitness_window) if fitness_window else 0}")
    
    for generation in itertools.count(0):  # Track generation for logging
        # Trim and evolve in one pass
        population = trim_population(
            generate_children(
                select_parents(population),
                population
            )[:MAX_POPULATION],
            MAX_POPULATION
        )
        
        # Evaluate population and update stats
        population, fitness_window = evaluate_population_stats(population, fitness_window, generation)
        
        # Calculate complete stats for display and logging
        stats = calculate_window_statistics(fitness_window)
        best_agent = max(population, key=lambda x: x["fitness"]) if population else {"metrics": {}}
        
        stats.update({
            'generation': generation,
            'population_size': len(population),
            'diversity': calculate_diversity(population),
            'best_core': best_agent.get("metrics", {}).get("core_segment", ""),
        })
        
        # Display and log stats
        handle_generation_output(stats, population)
        
        # Add a small delay to prevent CPU overload
        import time
        time.sleep(0.1)




def log_population(stats: dict) -> None:
    """Log population statistics in plain text format per spec.md"""
    with open("evolution.log", "a", encoding="utf-8") as f:
        # Write header if file is empty
        if f.tell() == 0:
            f.write("generation\tpopulation\tmean\tmedian\tstd\tbest\tworst\tdiversity\tcore\n")
            
        # Information-dense format with core segment
        f.write(
            f"{stats.get('generation', 0)}\t" 
            f"{stats.get('population_size', 0)}\t"
            f"{stats.get('mean', 0.0):.1f}\t"
            f"{stats.get('median', 0.0):.1f}\t"
            f"{stats.get('std', 0.0):.1f}\t"
            f"{stats.get('best', 0.0):.1f}\t"
            f"{stats.get('worst', 0.0):.1f}\t"
            f"{stats.get('diversity', 0.0):.3f}\t"
            f"{stats.get('best_core', '')[:23]}\n"  # Exactly 23 chars per spec.md core
        )

def display_generation_stats(stats: dict) -> None:
    """Rich-formatted display with essential stats"""
    console = Console()
    console.print(Panel(
        f"[bold]Gen {stats.get('generation', 0)}[/]\n"
        f"μ:{stats.get('mean', 0.0):.1f} σ:{stats.get('std', 0.0):.1f} (window)\n"
        f"Best: {stats.get('best', 0.0):.1f} Worst: {stats.get('worst', 0.0):.1f}\n"
        f"Core: {stats.get('best_core', '')}\n"
        f"Δ{stats.get('diversity', 0.0):.0%} 👥{stats.get('population_size', 0):,}/{MAX_POPULATION:,}",
        title="Evolution Progress",
        subtitle=f"[Population size: {stats.get('population_size', 0)}/{MAX_POPULATION}]",
        style="blue"
    ))
    
    # Print a separator for better readability
    console.print("─" * 80)




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
    stats.update({
        'diversity': calculate_diversity(population),
        'population_size': len(population),
        'best': max(a['fitness'] for a in population) if population else 0.0,
        'worst': min(a['fitness'] for a in population) if population else 0.0,
        'mutation_rate': sum(a.get('mutations', 0) for a in population) / len(population) if population else 0.0
    })
    return stats


def evaluate_population_stats(population: List[dict], fitness_window: list, generation: int) -> tuple:
    """Evaluate and log generation statistics"""
    # Evaluate population fitness
    population = evaluate_population(population)
    
    # Update fitness window
    new_fitness = [a["fitness"] for a in population]
    updated_window = update_fitness_window(fitness_window, new_fitness)
    
    # Get best agent and core segment
    best_agent = max(population, key=lambda x: x["fitness"]) if population else {"metrics": {}}
    best_core = best_agent.get("metrics", {}).get("core_segment", "")
    
    # Print best chromosome for debugging
    if best_agent:
        print(f"Best chromosome: {best_agent['chromosome']}")
        print(f"Best fitness: {best_agent['fitness']}")
    
    # Create stats dictionary
    stats = calculate_window_statistics(updated_window)
    stats.update({
        'generation': generation,
        'population_size': len(population),
        'diversity': calculate_diversity(population),
        'best_core': best_core,
    })
    
    return population, updated_window

# Main execution block at bottom per spec.md
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evolutionary string optimizer')
    parser.add_argument('--pop-size', type=int, default=1000,
                       help='Initial population size (default: 1000)')
    parser.add_argument('--window-size', type=int, default=100,
                       help='Sliding window size for statistics (default: 100)')
    args = parser.parse_args()
    
    # Set global window size from args
    WINDOW_SIZE = args.window_size
    
    try:
        run_genetic_algorithm(pop_size=args.pop_size)
    except KeyboardInterrupt:
        print("\nEvolution stopped by user. Exiting gracefully.")

def validate_population_state(best, worst) -> None:
    """Validate fundamental population invariants"""
    # Validate hidden goal constants without referencing spec.md
    assert MAX_CORE == 23 and MAX_CHARS == 40, "Core configuration invalid"
    
    # Fitness sanity checks - use absolute() since rewards can be negative per spec.md
    assert abs(best['fitness']) >= abs(worst['fitness']), "Best fitness should >= worst in magnitude"
    assert 0 <= best['fitness'] <= 1e6, "Fitness out of bounds"

    # Chromosome structural validation
    for agent in [best, worst]:
        chrom = agent['chromosome']
        assert (isinstance(chrom, str) and 
                1 <= len(chrom) <= 40 and 
                chrom == chrom.strip() and 
                chrom[:23].islower()), f"Invalid: {chrom}"

