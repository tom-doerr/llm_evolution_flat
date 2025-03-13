import argparse
import concurrent.futures
import random
import string
import sys
import time
from typing import List

# Third party imports

MAX_POPULATION = 1_000_000  # Defined per spec.md population limit
MAX_CHARS = 40  # From spec.md (different from max tokens)
MAX_CORE = 23  # From spec.md hidden goal
WINDOW_SIZE = 100  # Default, can be overridden by CLI

# Validate hidden goal constants from spec.md
assert MAX_CORE == 23, "Core segment length must be 23 per spec.md"
assert MAX_CHARS == 40, "Max chromosome length must be 40 for this task"

# Configure DSPy with OpenRouter and timeout
lm = dspy.LM(
    "openrouter/google/gemini-2.0-flash-001", max_tokens=80, timeout=10, cache=False
)
dspy.configure(lm=lm)
assert dspy.settings.lm is not None, "DSPy LM must be configured"

import sys

# Test mock configuration
if __name__ == "__main__" and "pytest" in sys.modules:  # pylint: disable=used-before-assignment
    lm = dspy.LM("mock_model")
    dspy.configure(lm=lm, test_mode=True)

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
    chromosome = _clean_input(chromosome)
    return _ensure_min_length(chromosome)

def _clean_input(chromosome: str) -> str:
    """Clean and normalize chromosome input"""
    if isinstance(chromosome, list):
        chromosome = "".join(chromosome)
    chrom = str(chromosome).strip()[:40].lower()
    return ''.join(c for c in chrom if c.isalpha() or c == ' ').strip()

def _ensure_min_length(chromosome: str) -> str:
    """Ensure minimum length and pad with 'a's if needed"""
    if not chromosome:
        return "a" * 23
    if len(chromosome) < 23:
        return chromosome.ljust(23, 'a')
    return chromosome

def create_agent(chromosome: str) -> dict:
    """Create agent with 3 specialized chromosomes"""
    chromosome = validate_chromosome(chromosome)
    
    # Pad chromosome to ensure all three segments exist
    padded_chromo = chromosome.ljust(40, random.choice(string.ascii_lowercase))
    
    return {
        "chromosome": padded_chromo,
        # Core task-solving instructions (first 23 chars)
        "task_chromosome": padded_chromo[:23].ljust(23, ' ')[:23],
        # Mate selection strategy (next 10 chars)
        "mate_selection_chromosome": padded_chromo[23:33].ljust(10, ' ')[:10].lower(),
        # Mutation strategy (final 7 chars)
        "mutation_chromosome": padded_chromo[33:40].ljust(7, ' ')[:7],
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
    penalty = len(chromo) - 23 if len(chromo) > 23 else 0
    
    # Calculate fitness
    agent["fitness"] = a_count - penalty
    
    # Add more detailed metrics for debugging
    metrics['a_count'] = a_count
    metrics['length_penalty'] = penalty
    
    assert len(metrics['core_segment']) == 23, "Core segment length mismatch"
    agent["metrics"] = metrics
    
    return agent["fitness"]


def initialize_population(pop_size: int) -> List[dict]:
    """Create initial population with varied 'a' density in core segment"""
    chromosomes = []
    
    # Create a diverse initial population with different 'a' densities
    for _ in range(pop_size):  # Fixed unused variable 'i'
        # Vary 'a' probability across population 
        a_probability = random.uniform(0.1, 0.5)  # 10-50% 'a's
        
        # Create core with controlled 'a' density
        core_chars = []
        for _ in range(23):
            if random.random() < a_probability:
                core_chars.append('a')
            else:
                core_chars.append(random.choice(string.ascii_lowercase))
        
        core = ''.join(core_chars)
        
        # Add minimal suffix (0-7 chars) to keep total length between 23-30
        suffix_len = random.randint(0, 7)
        suffix = ''.join(random.choices(string.ascii_lowercase, k=suffix_len))
        
        chromosomes.append(core + suffix)
    
    # Add one chromosome with all 'a's in core to seed the population
    if pop_size > 5:
        all_a_core = 'a' * 23
        chromosomes[0] = all_a_core
    
    # Create agents from chromosomes
    return [create_agent(c) for c in chromosomes]


def select_parents(population: List[dict]) -> List[dict]:
    """Select parents using Pareto(fitness²) weighting per spec.md"""
    # Implementation of spec.md required parent selection criteria:
    # 1. Pareto distribution weighting by fitness^2
    # 2. Weighted sampling without replacement
    if not population:
        return []
    
    # Weighted sampling per spec.md: fitness² * Pareto distribution
    # Pareto distribution weighting by fitness^2 per spec.md
    fitness_squared = np.array([max(a['fitness'], 0)**2 for a in population], dtype=np.float64)
    pareto = np.random.pareto(PARETO_SHAPE, len(population)) + 1
    weights = np.nan_to_num(fitness_squared * pareto, nan=1e-6).clip(1e-6)
    
    weights /= weights.sum()  # Normalize
    assert np.isclose(weights.sum(), 1.0), "Weights must sum to 1"
    assert len(weights) == len(population), "Weight/population size mismatch"
    
    # Weighted sampling without replacement using reservoir sampling
    selected_indices = []
    population_indices = list(range(len(population)))
    for _ in range(min(len(population), MAX_POPULATION//2)):
        chosen = random.choices(population_indices, weights=weights, k=1)[0]
        selected_indices.append(chosen)
        # Remove chosen index and reweight properly
        idx = population_indices.index(chosen)
        population_indices.pop(idx)
        weights = np.delete(weights, idx)
        if weights.sum() > 0:
            weights /= weights.sum()  # Renormalize
        else:  # Handle edge case
            weights = np.ones_like(weights) / len(weights)
    return [population[i] for i in selected_indices]

import numpy as np
import dspy
from rich.console import Console
from rich.panel import Panel

# Configuration constants from spec.md
PARETO_SHAPE = 3.0  # From spec.md parent selection requirements
MUTATION_RATE = 0.1  # Base mutation probability 
HOTSPOT_CHARS = {'.', ',', '!', '?', ';', ':', ' '}  # Expanded punctuation per spec.md
HOTSPOT_SPACE_PROB = 0.25  # Higher space probability per spec.md
MIN_HOTSPOTS = 2  # Ensure minimum 2 switch points for combination
HOTSPOT_ANYWHERE_PROB = 0.025  # 40 chars * 0.025 = 1 switch on average per spec.md
AVERAGE_SWITCHES = 1.0  # Explicit constant per spec.md requirement
HOTSPOT_ANYWHERE_PROB = 0.025  # 40 chars * 0.025 = 1 switch on average per spec.md



class MutateSignature(dspy.Signature):
    """Mutate chromosomes while preserving first 23 characters and increasing 'a' density."""
    chromosome = dspy.InputField(desc="Current chromosome to mutate")
    mutation_instructions = dspy.InputField(desc="Mutation strategy instructions") 
    mutated_chromosome = dspy.OutputField(desc="Improved chromosome meeting requirements")

def mutate_with_llm(agent: dict, cli_args: argparse.Namespace) -> str:
    """Optimized LLM mutation with validation"""
    agent["mutation_source"] = f"llm:{agent['mutation_chromosome']}"
    
    if cli_args.verbose:
        print(f"Attempting LLM mutation with instructions: {agent['mutation_chromosome']}")

    llm_result = _try_llm_mutation(agent, cli_args)
    return llm_result if llm_result else _fallback_mutation(agent)

def _try_llm_mutation(agent: dict, cli_args: argparse.Namespace) -> str:
    """Attempt LLM-based mutation and return valid result or None"""
    try:
        response = dspy.Predict(MutateSignature)(
            chromosome=agent["chromosome"],
            mutation_instructions=_build_mutation_prompt(agent)
        )
        return _process_llm_response(response, cli_args)
    except (dspy.DSPyError, ValueError) as e:
        if cli_args.verbose:
            print(f"LLM mutation error: {str(e)}")
        return None

def _build_mutation_prompt(agent: dict) -> str:
    """Construct mutation prompt string per spec.md requirements"""
    return f"""
    MUTATION INSTRUCTIONS: {agent['mutation_chromosome']}
    CURRENT CHROMOSOME: {agent["chromosome"]}
    REQUIREMENTS:
    - PRESERVE first 23 characters (core segment)
    - INCREASE 'a' density in first 23 characters
    - REDUCE length after 23 characters
    - USE ONLY lowercase letters and spaces
    OUTPUT ONLY the modified chromosome:
    """.strip()

def _process_llm_response(response, cli_args) -> str:
    """Process LLM response into valid chromosome"""
    for comp in response.completions:
        candidate = str(comp).strip().lower()[:MAX_CHARS]
        candidate = ''.join(c for c in candidate if c.isalpha() or c == ' ').strip()
        if len(candidate) >= MAX_CORE and validate_mutation(candidate):
            if cli_args.verbose:
                print(f"LLM mutation successful: {candidate}")
            return candidate
    return None

def _fallback_mutation(agent: dict, cli_args: argparse.Namespace) -> str:
    """Create fallback mutation with improved core"""
    core = list(agent["chromosome"][:MAX_CORE])
    a_count = core.count('a')
    
    # Add 'a's to core if needed
    if a_count < MAX_CORE:
        replacements = min(5, MAX_CORE - a_count)
        positions = random.sample([i for i, c in enumerate(core) if c != 'a'], replacements)
        for pos in positions:
            core[pos] = 'a'
    
    suffix = ''.join(random.choices(string.ascii_lowercase, k=random.randint(0, 7)))
    fallback = validate_chromosome(''.join(core) + suffix)
    
    if cli_args.verbose:
        print(f"Using fallback mutation: {fallback}")
    return fallback


def mutate(agent: dict, cli_args: argparse.Namespace) -> str:
    """Mutate a chromosome with LLM-based mutation as primary strategy"""
    mutated = mutate_with_llm(agent, cli_args)
    agent['mutations'] = agent.get('mutations', 0) + 1  # Track mutation count per agent
    return mutated


def validate_mutation(chromosome: str) -> bool:
    """Validate mutation meets criteria"""
    return (
        len(chromosome) >= 23 and
        len(chromosome) <= 40 and
        all(c.isalpha() or c == ' ' for c in chromosome) and  # From spec.md
        chromosome == chromosome.strip() and  # From spec.md
        chromosome[:23] == chromosome[:23].lower()  # Preserve core format
    )

def validate_mating_candidate(candidate: dict, parent: dict) -> bool:
    """Validate candidate meets mating requirements"""
    return all([
        candidate != parent,
        all(key in candidate for key in ("mutation_chromosome", "mate_selection_chromosome", "chromosome")),
        len(validate_chromosome(candidate["chromosome"])) >= 23,
        validate_chromosome(candidate["chromosome"]) != parent["chromosome"]
    ])

class MateSelectionSignature(dspy.Signature):
    """Select mate using agent's mate-selection chromosome as instructions"""
    mate_selection_chromosome = dspy.InputField(desc="Mate-selection chromosome/prompt of parent agent") 
    parent_dna = dspy.InputField(desc="DNA of parent agent selecting mate")
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
        parent_dna=parent["chromosome"],
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
        
    # Calculate target number of switches based on spec.md average requirement
    target_switches = max(1, round(len(chromosome) * AVERAGE_SWITCHES / 40))
    
    # First collect punctuation and space-based hotspots
    hotspots = [
        i for i, c in enumerate(chromosome)
        if c in HOTSPOT_CHARS or (c == ' ' and random.random() < HOTSPOT_SPACE_PROB)
    ]
    
    # Then add random hotspots to reach target average
    while len(hotspots) < target_switches:
        pos = random.randint(0, len(chromosome)-1)
        if pos not in hotspots:
            hotspots.append(pos)
    
    # Ensure minimum hotspots per spec
    if len(hotspots) < MIN_HOTSPOTS and chromosome:
        hotspots.extend(random.sample(range(len(chromosome)), k=MIN_HOTSPOTS-len(hotspots)))
    
    return sorted(list(set(hotspots)))  # Remove duplicates and sort

def build_child_chromosome(parent: dict, mate: dict) -> str:
    """Construct child chromosome with switches at hotspots"""
    p_chrom = parent["chromosome"]
    m_chrom = mate["chromosome"]
    
    hotspots = get_hotspots(p_chrom)
    result = []
    use_parent = True
    last_pos = 0

    for pos in sorted(hotspots):
        if pos >= len(p_chrom):
            continue
            
        # Take segment from current parent
        result.append(p_chrom[last_pos:pos] if use_parent else m_chrom[last_pos:pos])
        use_parent = not use_parent
        last_pos = pos

    # Add remaining sequence
    result.append(p_chrom[last_pos:] if use_parent else m_chrom[last_pos:])
    
    return validate_chromosome("".join(result))

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

def run_evolution(population_size: int = 1000) -> list:
    """Run evolutionary optimization"""
    population = initialize_population(min(population_size, MAX_POPULATION))[:MAX_POPULATION]
    evolution_loop(population)
    return population

def run_genetic_algorithm(pop_size: int) -> None:
    """Run continuous genetic algorithm per spec.md"""
    population = initialize_population(min(pop_size, MAX_POPULATION))[:MAX_POPULATION]
    assert 1 < len(population) <= MAX_POPULATION, f"Population size must be 2-{MAX_POPULATION}"
    
    # Initialize log with header and truncate any existing content per spec.md
    with open("evolution.log", "w", encoding="utf-8") as f:
        pass  # File is emptied by opening in write mode
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
    current_stats = calculate_window_statistics(fitness_values)
    stats = calculate_window_statistics(window)
    stats.update({
        'current_mean': current_stats['mean'],
        'current_median': current_stats['median'], 
        'current_std': current_stats['std']
    })
    
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

def evolution_loop(population: List[dict], args: argparse.Namespace) -> None:
    """Continuous evolution loop without discrete generations"""
    
    fitness_window = []
    num_threads = args.threads
    iterations = 0
    
    # Initial evaluation of population
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_agent = {executor.submit(evaluate_agent, agent): agent for agent in population}
        for future in concurrent.futures.as_completed(future_to_agent):
            agent = future_to_agent[future]
            try:
                agent["fitness"] = future.result()
            except Exception as e:
                print(f"Agent evaluation failed: {str(e)}")
                raise RuntimeError("Population evaluation failed") from e
    
    fitness_window = [a["fitness"] for a in population]
    
    # Print initial stats
    print(f"Initial population: {len(population)} agents")
    print(f"Initial best fitness: {max(fitness_window) if fitness_window else 0}")
    print("Starting continuous evolution...")
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            while True:  # Continuous evolution without generations
                iterations += 1
                
                # Select parents based on fitness
                selected_parents = select_parents(population)
                
                # Randomly select a parent for reproduction
                parent = random.choice(selected_parents)
                
                # Either crossover or mutate
                if random.random() < 0.9 and len(population) > 1:
                    # Submit crossover task
                    future = executor.submit(crossover, parent, population)
                else:
                    # Submit mutation task
                    future = executor.submit(lambda p: create_agent(mutate(p)), parent)
                
                try:
                    # Get the new child
                    child = future.result()
                    
                    # Evaluate the child
                    child["fitness"] = evaluate_agent(child)
                    
                    # Add to population
                    population.append(child)
                    
                    # Trim population if needed
                    if len(population) > MAX_POPULATION:
                        population = trim_population(population, MAX_POPULATION)
                    
                    # Update fitness window
                    fitness_window = update_fitness_window(fitness_window, [child["fitness"]])
                    
                except (ValueError, TypeError) as e:
                    print(f"Child creation failed: {e}")
                
                # Display stats periodically
                if iterations % 10 == 0:
                    stats = calculate_window_statistics(fitness_window)
                    best_agent = max(population, key=lambda x: x["fitness"]) if population else {"metrics": {}}
                    
                    stats.update({
                        'generation': iterations,  # Use iterations instead of generations
                        'population_size': len(population),
                        'diversity': calculate_diversity(population),
                        'best_core': best_agent.get("metrics", {}).get("core_segment", ""),
                    })
                    
                    # Display and log stats
                    handle_generation_output(stats, population)
                    
                    # Print best chromosome for debugging
                    if args.verbose and best_agent:
                        print(f"Best chromosome: {best_agent['chromosome']}")
                        print(f"Best fitness: {best_agent['fitness']}")
                        print(f"A's in core: {best_agent['chromosome'][:23].count('a')}")
                        print(f"Length after core: {len(best_agent['chromosome']) - 23}")
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nEvolution stopped by user. Exiting gracefully.")




def log_population(stats: dict) -> None:
    """Log population statistics in plain text format per spec.md"""
    with open("evolution.log", "a", encoding="utf-8") as f:
        # Information-dense format with core segment
        f.write(f"{stats.get('generation', 0)}\t"
                f"{stats.get('population_size', 0)}\t"
                f"{stats.get('current_mean', 0.0):.1f}\t"
                f"{stats.get('current_median', 0.0):.1f}\t"
                f"{stats.get('current_std', 0.0):.1f}\t"
                f"{stats.get('best', 0.0):.1f}\t"
                f"{stats.get('worst', 0.0):.1f}\t"
                f"{stats.get('diversity', 0.0):.2f}\t"
                f"{stats.get('best_core', '')[:23]}\n")

def display_generation_stats(stats: dict) -> None:
    """Rich-formatted display with essential stats"""
    console = Console()
    
    # Get the best agent's core and count 'a's
    best_core = stats.get('best_core', '')
    a_count = best_core.count('a') if best_core else 0
    
    console.print(Panel(
        f"[bold]Gen {stats.get('generation', 0)}[/]\n"
        f"Current μ:{stats.get('current_mean', 0.0):.1f} σ:{stats.get('current_std', 0.0):.1f}\n"
        f"Window μ:{stats.get('window_mean', 0.0):.1f} σ:{stats.get('window_std', 0.0):.1f}\n"
        f"Best: {stats.get('best', 0.0):.1f} Worst: {stats.get('worst', 0.0):.1f}\n"
        f"Core: [a's:{a_count}/23] {best_core[:10]}...{best_core[-10:]}\n" 
        f"Δ{stats.get('diversity', 0.0):.0%} 👥{stats.get('population_size', 0):,}/{MAX_POPULATION:,}",
        title="Evolution Progress",
        subtitle=f"[Population: {stats.get('population_size', 0):,}/{MAX_POPULATION:,}]",
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
    return len({a["chromosome"] for a in population}) / len(population) if population else 0.0





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
    print("Evaluating population fitness...")
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
        print(f"A's in core: {best_agent['chromosome'][:23].count('a')}")
        print(f"Length after core: {len(best_agent['chromosome']) - 23}")
    
    # Create stats dictionary
    stats = calculate_window_statistics(updated_window)
    stats.update({
        'generation': generation,
        'population_size': len(population),
        'diversity': calculate_diversity(population),
        'best_core': best_core,
    })
    
    return population, updated_window

def validate_population_state(best, worst) -> None:
    """Validate fundamental population invariants"""
    # Validate hidden goal constants without referencing spec.md
    assert MAX_CORE == 23 and MAX_CHARS == 40, "Core configuration invalid"
    
    # Fitness sanity checks - use absolute() since rewards can be negative per spec.md
    assert best['fitness'] >= worst['fitness'], "Best fitness should be >= worst fitness"
    
    # Chromosome structural validation
    for agent in [best, worst]:
        chrom = agent['chromosome']
        assert (isinstance(chrom, str) and 
                1 <= len(chrom) <= 40 and 
                chrom == chrom.strip()), f"Invalid chromosome: {chrom}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evolutionary string optimizer')
    parser.add_argument('--pop-size', type=int, default=1000,
                       help='Initial population size (default: 1000)')
    parser.add_argument('--window-size', type=int, default=100,
                       help='Sliding window size for statistics (default: 100)')
    parser.add_argument('--threads', type=int, default=10,
                       help='Number of parallel threads (default: 10)',
                       choices=range(1, 21))  # Limit 1-20 threads per spec
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    args = parser.parse_args()
    
    # Set global window size from args
    WINDOW_SIZE = args.window_size
    
    try:
        run_genetic_algorithm(args.pop_size)
    except KeyboardInterrupt:
        print("\nEvolution stopped by user. Exiting gracefully.")

