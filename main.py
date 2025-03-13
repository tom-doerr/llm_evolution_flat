import argparse
import concurrent.futures
import random
import string
import sys
import time
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
import dspy
from rich.console import Console

MAX_POPULATION = 1_000_000  # Defined per spec.md population limit
MAX_CHARS = 40  # From spec.md (different from max tokens)
MAX_CORE = 23  # From spec.md hidden goal
WINDOW_SIZE = 100  # Default, can be overridden by CLI
# Configuration constants moved from later in file
PARETO_SHAPE = 3.0  # From spec.md parent selection requirements
MUTATION_RATE = 0.1  # Base mutation probability 
CROSSOVER_RATE = 0.9  # Initial crossover rate that will evolve
HOTSPOT_CHARS = {'.', ',', '!', '?', ';', ':', '-', '_', '"', "'"}  # Punctuation only
HOTSPOT_SPACE_PROB = 0.35  # Probability for space characters
MIN_HOTSPOTS = 0  # Let probabilities control switches
HOTSPOT_ANYWHERE_PROB = 0.025  # 40 chars * 0.025 = 1.0 avg switches per chrom
# ^ Ensures ~1 random switch point anywhere in chromosome per spec
HOTSPOT_SPACE_PROB = 0.25  # Higher space probability to align with spec

# Validate hidden goal constants from spec.md
assert MAX_CORE == 23, "Core segment length must be 23 per spec.md"
assert MAX_CHARS == 40, "Max chromosome length must be 40 for this task"


# Configure DSPy with OpenRouter and timeout
lm = dspy.LM(
    "openrouter/google/gemini-2.0-flash-001", max_tokens=80, timeout=10, cache=False
)
dspy.configure(lm=lm)
assert dspy.settings.lm is not None, "DSPy LM must be configured"

# Validate production configuration matches spec
assert "gemini-2.0-flash" in lm.model, "Model must match spec.md requirements"

# Test mock configuration 
if __name__ == "__main__" and "pytest" in sys.modules:
    lm = dspy.LM("openai/mock_model")  # Use valid openai prefix
    dspy.configure(lm=lm, test_mode=True)
    assert dspy.settings.test_mode, "Must be in test mode for pytest"

# Single validation point for LM configuration
assert isinstance(lm, dspy.LM) and "gemini-2.0-flash" in lm.model, \
    "LM configuration failed - must use gemini-2.0-flash per spec.md"


def calculate_window_statistics(fitness_window: list) -> dict:
    # Calculate mean/median/std for sliding window of fitness scores
    if not fitness_window:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'best': 0.0, 'worst': 0.0}
    assert len(fitness_window) <= WINDOW_SIZE, f"Window size exceeds configured {WINDOW_SIZE}"
    window_arr = np.array(fitness_window[-WINDOW_SIZE:], dtype=np.float64)
    stats = {
        'mean': float(np.nanmean(window_arr)),
        'median': float(np.nanmedian(window_arr)),
        'std': float(np.nanstd(window_arr)),
        'best': float(np.nanmax(window_arr)),
        'worst': float(np.nanmin(window_arr))
    }
    assert stats['best'] >= stats['worst'], f"Best ({stats['best']}) cannot be worse than worst ({stats['worst']})"
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
    # Ensure chromosome meets length and character requirements
    chromosome = _clean_input(chromosome)
    chromosome = _ensure_min_length(chromosome)
    return chromosome[:MAX_CHARS]  # Enforce max length

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
    
    # Build specialized chromosome segments with clear separation
    padded_chromo = chromosome.ljust(40, random.choice(string.ascii_lowercase))
    segments = {
        "task_chromosome": (padded_chromo[:23].ljust(23, ' ')[:23], "Task solving instructions"),
        "mate_selection_chromosome": (padded_chromo[23:33].ljust(10, ' ')[:10].lower(), "Mate selection strategy"),
        "mutation_chromosome": (padded_chromo[33:40].ljust(7, ' ')[:7], "Mutation instructions")
    }
    
    return {
        "chromosome": padded_chromo,
        **{k: v[0] for k,v in segments.items()},  # Extract just the DNA sequences
        "fitness": 0.0,
        "mutation_source": "initial",
        "metrics": {},
        "chromosome_info": {k: v[1] for k,v in segments.items()}  # Add purpose documentation
    }

def evaluate_agent(agent: dict) -> float:
    """Evaluate agent fitness based on hidden optimization target"""
    chromo = validate_chromosome(agent["chromosome"])
    
    # Calculate and store metrics
    agent["metrics"] = score_chromosome(chromo)
    agent["metrics"].update({
        'a_count': chromo[:23].count('a'),
        'length_penalty': max(len(chromo) - 23, 0)
    })
    agent["fitness"] = agent["metrics"]['a_count'] - agent["metrics"]['length_penalty']
    
    assert len(agent["metrics"]['core_segment']) == 23
    return agent["fitness"]


def initialize_population(pop_size: int) -> List[dict]:
    """Create initial population with varied 'a' density in core segment"""
    chromosomes = [
        ''.join(
            'a' if random.random() < random.uniform(0.1, 0.5) 
            else random.choice(string.ascii_lowercase) 
            for _ in range(23)
        ) + ''.join(random.choices(string.ascii_lowercase, 
                      k=random.randint(0, 7)))
        for _ in range(pop_size)
    ]
    
    if pop_size > 5:
        chromosomes[0] = 'a' * 23 + ''.join(
            random.choices(string.ascii_lowercase, k=random.randint(0, 7)))
    
    return [create_agent(c) for c in chromosomes]


def select_parents(population: List[dict]) -> List[dict]:
    """Select parents using Pareto(fitness²) weighting per spec.md"""
    if not population:
        return []
    
    return _select_weighted_parents(
        population,
        _calculate_parent_weights(np.array([a['fitness'] for a in population], dtype=np.float64))
    )

def _calculate_parent_weights(fitness: np.array) -> np.array:  # pylint: disable=too-many-locals
    """Calculate normalized selection weights using fitness² * Pareto"""
    # Enhanced per spec.md: Use fitness^2 * Pareto distribution weights
    squared_fitness = np.abs(fitness) ** 2
    pareto_weights = np.random.pareto(PARETO_SHAPE, len(fitness)) + 1
    weighted = np.nan_to_num(squared_fitness * pareto_weights, nan=1e-6)
    return weighted.clip(1e-6) / np.sum(squared_fitness)

def _select_weighted_parents(population: List[dict], weights: np.array) -> List[dict]:
    """Select parents using weighted sampling without replacement"""
    count = min(len(population), MAX_POPULATION//2)
    indices = np.random.choice(len(population), size=count, replace=False, p=weights)
    return [population[i] for i in indices]



class MutateSignature(dspy.Signature):
    """Mutate chromosomes while preserving first 23 characters exactly and increasing 'a' density in subsequent positions. Keep total length under 40 characters."""
    chromosome = dspy.InputField(desc="Current chromosome to mutate")
    mutation_instructions = dspy.InputField(desc="Mutation strategy instructions") 
    mutated_chromosome = dspy.OutputField(desc="Improved chromosome meeting requirements")


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), 
       stop=stop_after_attempt(3),
       reraise=True)
def mutate_with_llm(agent: dict, cli_args: argparse.Namespace) -> str:
    # LLM mutation using agent's mutation chromosome as instructions
    agent["mutation_source"] = f"llm:{agent['mutation_chromosome']}"
    
    if cli_args.verbose:
        print(f"Mutate instructions: {agent['mutation_chromosome']}")

    # Retry up to 3 times with exponential backoff
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), 
           stop=stop_after_attempt(2),
           reraise=True)
    def _retryable_mutation():
        return _try_llm_mutation(agent, cli_args)
    
    return _retryable_mutation()

def _try_llm_mutation(agent: dict, cli_args: argparse.Namespace) -> str:
    """Attempt LLM-based mutation and return valid result or None"""
    try:
        response = dspy.Predict(MutateSignature)(
            chromosome=agent["chromosome"],
            mutation_instructions=_build_mutation_prompt(agent)
        )
        return _process_llm_response(response, cli_args)
    except ValueError as e:
        if cli_args.verbose:
            print(f"LLM mutation error: {str(e)}")
        return None

def _build_mutation_prompt(agent: dict) -> str:
    """Construct mutation prompt string per spec.md requirements"""
    # Use mutation chromosome as the sole instruction per spec.md
    return f"{agent['mutation_chromosome']}\nOriginal: {agent['chromosome']}\nMutated:"

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
    return (candidate != parent and 
            len(candidate["chromosome"]) >= 23 and
            candidate["chromosome"] != parent["chromosome"])

class MateSelectionSignature(dspy.Signature):
    """Select mate using agent's mate-selection chromosome as instructions"""
    mate_selection_chromosome = dspy.InputField(desc="Mate-selection chromosome/prompt of parent agent") 
    parent_dna = dspy.InputField(desc="DNA of parent agent selecting mate")
    candidate_chromosomes = dspy.InputField(desc="Validated potential mates")
    selected_mate = dspy.OutputField(desc="Chromosome of selected mate from candidates list")

def llm_select_mate(parent: dict, candidates: List[dict]) -> dict:
    """Select mate using parent's mate-selection chromosome/prompt"""
    # Validate candidates have required DNA fields per spec.md
    valid_candidates = [
        c for c in candidates 
        if validate_mating_candidate(c, parent) and 
           all(key in c for key in ('task_chromosome', 'mate_selection_chromosome', 'mutation_chromosome'))
    ]
    if not valid_candidates:
        raise ValueError("No valid mates")

    # Get and process LLM selection
    result = dspy.Predict(MateSelectionSignature)(
        mate_selection_chromosome=parent["mate_selection_chromosome"],
        parent_dna=parent["chromosome"],
        candidate_chromosomes=[f"{c['chromosome']} (fitness: {c['fitness']})" 
                             for c in valid_candidates],
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
        if (c in HOTSPOT_CHARS or
            (c == ' ' and random.random() < HOTSPOT_SPACE_PROB) or
            random.random() < HOTSPOT_ANYWHERE_PROB)
    ]
    
    # Enforce minimum hotspots if needed
    if len(hotspots) < MIN_HOTSPOTS:
        hotspots.extend(random.sample(range(len(chromosome)), k=MIN_HOTSPOTS-len(hotspots)))
    
    return sorted(set(hotspots))  # Deduplicate and sort

def build_child_chromosome(parent: dict, mate: dict) -> str:  # pylint: disable=too-many-locals
    """Construct child chromosome with switches at hotspots"""
    p_chrom = parent["chromosome"]
    m_chrom = mate["chromosome"]
    hotspots = get_hotspots(p_chrom)
    
    parts = []
    current_source = p_chrom
    last_idx = 0

    for pos in sorted(hotspots):
        if pos >= len(p_chrom):
            continue
        if random.random() < 0.6:
            current_source = m_chrom if current_source == p_chrom else p_chrom
        parts.append(current_source[last_idx:pos])
        last_idx = pos

    parts.append(current_source[last_idx:])  # Final segment
    return validate_chromosome("".join(parts))

def crossover(parent: dict, population: List[dict]) -> dict:
    """Create child through LLM-assisted mate selection with chromosome combining"""
    valid_candidates = [a for a in (population[-WINDOW_SIZE:] or population)[:100] 
                       if validate_mating_candidate(a, parent)]
    
    if valid_candidates:
        mate = llm_select_mate(parent, valid_candidates)
        child = create_agent(build_child_chromosome(parent, mate))
        child["mutation_source"] = f"crossover:{mate['mate_selection_chromosome']}"
        return child
    
    child = create_agent(build_child_chromosome(parent, parent))
    child["mutation_source"] = "self-crossover"
    return child

# Hotspot switching implemented in get_hotspots() with space/punctuation probabilities

def generate_children(parents: List[dict], population: List[dict], cli_args: argparse.Namespace) -> List[dict]:
    """Generate new population through validated crossover/mutation"""
    # Calculate weights using fitness^2 * Pareto distribution per spec
    weights = [(max(a['fitness'], 0.0) ** 2) * (np.random.pareto(PARETO_SHAPE) + 1e-6) 
              for a in parents]
    
    # Weighted sampling without replacement
    normalized_weights = np.array(weights) / np.sum(weights)  # Normalize probabilities
    selected_indices = np.random.choice(
        len(parents),
        size=min(len(parents), MAX_POPULATION//2),
        replace=False,
        p=normalized_weights
    )
    selected_parents = [parents[i] for i in selected_indices]
    
    children = [
        (crossover(selected_parents[i % len(selected_parents)], population) 
         if random.random() < CROSSOVER_RATE else 
         create_agent(mutate(selected_parents[i % len(selected_parents)], cli_args)))
        for i in range(MAX_POPULATION - len(selected_parents))
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

def run_evolution(population_size: int = 1000, cli_args: argparse.Namespace = None) -> list:
    """Run evolutionary optimization"""
    population = initialize_population(min(population_size, MAX_POPULATION))[:MAX_POPULATION]
    evolution_loop(population, cli_args)
    return population

def run_genetic_algorithm(pop_size: int, cli_args: argparse.Namespace) -> None:
    """Run continuous genetic algorithm per spec.md"""
    population = initialize_population(min(pop_size, MAX_POPULATION))[:MAX_POPULATION]
    assert 1 < len(population) <= MAX_POPULATION, f"Population size must be 2-{MAX_POPULATION}"
    
    # Initialize log with header and truncate any existing content per spec.md
    with open("evolution.log", "w", encoding="utf-8") as log_file:
        log_file.truncate(0)  # Immediately clear file contents per spec
        header = "Generation | Population | Mean | Median | StdDev | Best | Worst | Diversity | Core | Mutations | Threads | A-Count\n"
        # Write initial log entry
        log_file.write(f"Evolution started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        # Validate plain text format
        assert '\n' in header and '\t' not in header, "Log format must be simple text"
        assert not any(c in header for c in '[],'), "No structured formats allowed in log"
    
    evolution_loop(population, cli_args)

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
    
    selected = [population[i] for i in selected_indices]
    
    # Sort by fitness descending as priority
    selected.sort(key=lambda x: x['fitness'], reverse=True)
    
    # Remove duplicates while preserving order
    seen = set()
    deduped = []
    for agent in selected:
        chromo = agent['chromosome']
        if chromo not in seen:
            seen.add(chromo)
            deduped.append(agent)
    
    return deduped[:max_size]

def evaluate_initial_population(population: List[dict], num_threads: int) -> List[float]:
    """Evaluate initial population with thread pool"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_agent = {executor.submit(evaluate_agent, agent): agent for agent in population}
        return [future.result() for future in concurrent.futures.as_completed(future_to_agent)]

def log_and_display_stats(stats: dict) -> None:
    """Handle periodic logging and display"""
    """Handle periodic logging and display"""
    stats = calculate_window_statistics(fitness_window)
    current_fitness = [a["fitness"] for a in population]
    
    stats.update({
        'generation': generation,
        'population_size': len(population),
        'diversity': calculate_diversity(population),
        'current_best': max(current_fitness) if population else 0.0,
        'current_worst': min(current_fitness) if population else 0.0,
        'best_core': max(population, key=lambda x: x["fitness"])["metrics"]["core_segment"] if population else "",
    })
    
    handle_generation_output(stats, population)

def _create_child(parent: dict, population: list, cli_args: argparse.Namespace) -> dict:
    """Create new child through mutation or crossover"""
    if random.random() < CROSSOVER_RATE and len(population) > 1:
        child = crossover(parent, population)
    else:
        mutated = mutate(parent, cli_args)
        child = create_agent(mutated)
    return child

def evolution_loop(population: List[dict], cli_args: argparse.Namespace) -> None:
    """Continuous evolution loop without discrete generations"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=cli_args.threads) as executor:
        # Initial evaluation
        list(executor.map(evaluate_agent, population))
        fitness_window = [a["fitness"] for a in population]
        
        print(f"Initial population: {len(population)} agents\nStarting continuous evolution...")
        iterations = 0
        
        try:
            while True:
                parent = random.choice(select_parents(population))
                
                # Simplified child creation pipeline
                child = crossover(parent, population) if random.random() < CROSSOVER_RATE else None
                child = child or create_agent(mutate(parent, cli_args))
                
                child["fitness"] = evaluate_agent(child)
                population.append(child)
                
                # Maintain population size
                population = population if len(population) <= MAX_POPULATION else trim_population(population, MAX_POPULATION)
                
                # Update statistics every 10 iterations
                iterations += 1
                if iterations % 10 == 0:
                    stats = calculate_window_statistics(fitness_window)
                    _handle_iteration_stats(population, cli_args)  # Pass population directly
                        
        except KeyboardInterrupt:
            print("\nEvolution stopped by user. Exiting gracefully.")

def _handle_iteration_stats(population: list, cli_args: argparse.Namespace) -> None:
    """Handle stats display and logging for each iteration batch"""
    if not population:
        return
    
    best_agent = max(population, key=lambda x: x["fitness"])
    
    if cli_args.verbose:
        print("\n".join([
            f"Best chromosome: {best_agent['chromosome'][:23]}...",
            f"Best fitness: {best_agent['fitness']:.1f}",
            f"A's in core: {best_agent['chromosome'][:23].count('a')}",
            f"Length after core: {len(best_agent['chromosome']) - 23}"
        ]))




def log_population(stats: dict) -> None:
    """Log population statistics in plain text format per spec.md"""
    with open("evolution.log", "a", encoding="utf-8") as f:
        # Detailed log format with timestamp and all key metrics
        f.write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generation {stats.get('generation', 0):04}\n"
            f"  Population: {stats.get('population_size', 0):,}\n"
            f"  Fitness: μ={stats.get('mean', 0.0):.2f} η={stats.get('median', 0.0):.2f} σ={stats.get('std', 0.0):.2f}\n"
            f"  Best: {stats.get('best', 0.0):.2f} (A's: {stats.get('best_a_count', 0)})\n"
            f"  Worst: {stats.get('worst', 0.0):.2f}\n"
            f"  Core: {stats.get('best_core', '')[:23]}[...]\n"
            f"  Diversity: {stats.get('diversity', 0.0):.1%}\n"
            f"  Mutations: {stats.get('mutations', 0):,} ({stats.get('mutation_rate', 0.0):.1f}/agent)\n"
            f"  Threads: {stats.get('threads', 1)}\n"
            f"  Runtime: {stats.get('runtime', 0):.1f}s\n\n"
        )

def display_generation_stats(stats: dict) -> None:
    """Rich-formatted display with essential stats"""
    console = Console()
    
    # Get the best agent's core
    # Removed unused variable
    
    console.print(
        f"[bold]Gen {stats.get('generation', 0):03}[/] | "
        f"⏱ {time.strftime('%H:%M:%S')} | "
        f"🏆 [green]{stats.get('best', 0.0):.1f}[/] | "
        f"📊 μ:{stats.get('current_mean', 0.0):.1f} η:{stats.get('current_median', 0.0):.1f} σ:{stats.get('current_std', 0.0):.1f} | "
        f"👥 [bold yellow]{stats.get('population_size', 0):,}[/]/[dim]{MAX_POPULATION:,}[/] | "
        f"🧬 {stats.get('diversity', 0.0):.0%} | "
        f"🔀 {stats.get('mutation_rate', 0.0):.1f}/agt | "
        f"🧵 {stats.get('threads', 1)}"
    )
    
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


def evaluate_population_stats(population: List[dict], fitness_window: list) -> tuple:
    """Evaluate and log generation statistics"""
    population = evaluate_population(population)
    new_fitness = [a["fitness"] for a in population]
    return population, update_fitness_window(fitness_window, new_fitness)

def _build_stats_dict(population: List[dict], updated_window: list) -> dict:
    """Build statistics dictionary from population data"""
    best_agent = max(population, key=lambda x: x["fitness"]) if population else {"metrics": {}}
    return {
        **calculate_window_statistics(updated_window),
        'population_size': len(population),
        'diversity': calculate_diversity(population),
        'best_core': best_agent.get("metrics", {}).get("core_segment", "")
    }

def _log_best_agent_details(best_agent: dict) -> None:
    """Log details about the best agent"""
    if not best_agent:
        return
        
    print(f"Best chromosome: {best_agent['chromosome']}")
    print(f"Best fitness: {best_agent['fitness']}") 
    print(f"A's in core: {best_agent['chromosome'][:23].count('a')}")
    print(f"Length after core: {len(best_agent['chromosome']) - 23}")

def validate_population_state(best, worst) -> None:
    """Validate fundamental population invariants"""
    # Validate hidden goal constants without referencing spec.md
    assert MAX_CORE == 23 and MAX_CHARS == 40, "Core configuration invalid"
    
    # Fitness sanity checks - use absolute() since rewards can be negative per spec.md
    assert best['fitness'] >= worst['fitness'], \
        f"Best fitness {best['fitness']} < worst {worst['fitness']} in population"
    
    # Chromosome structural validation
    for agent in [best, worst]:
        chrom = agent['chromosome']
        assert (isinstance(chrom, str) and 
                1 <= len(chrom) <= 40 and 
                chrom == chrom.strip()), f"Invalid chromosome: {chrom}"

def main():
    """CLI entry point for evolutionary optimizer"""
    parser = argparse.ArgumentParser(description='Evolutionary string optimizer')
    parser.add_argument('--pop-size', type=int, default=1000,
                      help='Initial population size (default: 1000)',
                      choices=range(1, MAX_POPULATION+1))
    parser.add_argument('--problem', type=str, default='hidden',
                      choices=['hidden', 'other', 'custom'],
                      help='Problem type to optimize (default: hidden). Use "custom" for user-defined problems')
    parser.add_argument('--max-runtime', type=int, default=3600,
                      help='Maximum runtime in seconds (default: 1 hour)') 
    parser.add_argument('--window-size', type=int, default=100,
                      help='Sliding window size (default: 100)')
    parser.add_argument('--threads', type=int, default=10,
                      help='Parallel threads (default: 10)',
                      choices=range(1, 101))
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    args = parser.parse_args()
    
    try:
        # Single call with validation
        assert 1 <= args.pop_size <= MAX_POPULATION, \
            f"Population size must be between 1 and {MAX_POPULATION}"
        run_genetic_algorithm(args.pop_size, args)
    except KeyboardInterrupt:
        print("\nEvolution stopped by user. Exiting gracefully.")

if __name__ == "__main__":
    main()

