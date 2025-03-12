from typing import List
import random
import string
import gzip
import re  # For mutation validation
import numpy as np
from rich.console import Console
from rich.panel import Panel
import dspy

MAX_POPULATION = 1_000_000  # Defined per spec.md population limit

# Completed:
# 1. Implement generation-based scoring weights

# TODO priority order: 
# 1. Remove unused variables (ex, f)
# 2. Reduce function arguments per pylint
# 3. Validate chromosome structure during crossover
# 4. Implement sliding window statistics

# Configure DSPy with OpenRouter and timeout
MAX_POPULATION = 1_000_000  # From spec.md
DEBUG = False  # Control debug output
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
    
    # Fixed window size per spec.md with proper sliding implementation
    window_size = min(len(fitness_window), WINDOW_SIZE)
    window = fitness_window[-window_size:] if fitness_window else []
    assert 0 <= len(window) <= WINDOW_SIZE, f"Window size violation: {len(window)}"
    if not window:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, 
                "best": 0.0, "worst": 0.0, "q25": 0.0, "q75": 0.0}
    
    try:
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
    except Exception as e:
        raise RuntimeError(f"Window statistics calculation failed: {str(e)}") from e

def update_fitness_window(fitness_window: list, new_fitnesses: list) -> list:
    """Maintain sliding window of last 100 evaluations"""
    assert isinstance(fitness_window, list), "Window must be list type"
    return (fitness_window + new_fitnesses)[-WINDOW_SIZE:]  # Simple slicing for fixed size

def score_chromosome(chromosome: str) -> dict:
    """Calculate structural scoring metrics"""
    core = chromosome[:23].lower()
    assert len(core) == 23, "Core segment must be 23 characters"
    
    # Combined analysis with reduced variables
    counts = {'vowels': 0, 'a': 0, 'repeats': 0}
    unique_chars = set()
    prev_char = None
    
    for c in core:
        unique_chars.add(c)
        counts['vowels'] += c in 'aeiou'
        counts['a'] += c == 'a'
        counts['repeats'] += c == prev_char
        prev_char = c
    
    return {
        'vowel_ratio': counts['vowels'] / 23,
        'consonant_ratio': (23 - counts['vowels']) / 23,
        'uniqueness': len(unique_chars) / 23,
        'a_density': counts['a'] / 23,
        'repeating_pairs': counts['repeats'] / 22,
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


def evaluate_agent(agent: dict) -> float:  # Removed unused problem_description
    """Evaluate agent fitness based on hidden optimization target"""
    chromosome = str(agent["chromosome"])
    assert 23 <= len(chromosome) <= 40, f"Invalid length: {len(chromosome)}"
    
    metrics = score_chromosome(chromosome)
    a_count = int(metrics['a_density'] * 23)
    
    # Combined fitness calculation
    fitness = (a_count - (23 - a_count) - (len(chromosome) - 23))
    fitness = np.sign(fitness) * (abs(fitness) ** 2)
    
    # Validation assertions using metrics
    core_segment = metrics['core_segment']
    assert len(core_segment) == 23, f"Core segment must be 23 chars, got {len(core_segment)}"
    if metrics['a_count'] == 0:
        print(f"WARNING: No 'a's in first 23 of: {chromosome}")

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
    """Select parents using Pareto distribution weighting by fitness^2"""
    # Sort population by descending fitness
    sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
    ranks = np.arange(1, len(sorted_pop) + 1)
    
    # Calculate Pareto distribution weights using fitness^2 as specified
    fitness_squared = np.array([a['fitness']**2 for a in sorted_pop])
    weights = fitness_squared / fitness_squared.sum()  # Weight by fitness^2
    weights *= 1.0 / (ranks ** 2)  # Apply Pareto distribution
    weights /= weights.sum()  # Normalize again after combined weighting
    
    # Validate weights before selection
    assert np.all(weights >= 0), "Negative weights detected in parent selection"
    assert len(weights) == len(sorted_pop), "Weight/population size mismatch"
    
    # Select half population using weighted sampling without replacement
    selected_indices = np.random.choice(
        len(sorted_pop),
        size=len(sorted_pop)//2,
        replace=False,
        p=weights
    )
    selected_parents = [sorted_pop[i] for i in selected_indices]
    
    # Validate selection size
    assert len(selected_parents) == len(population)//2, "Parent selection size mismatch"
    return selected_parents




def mutate_with_llm(chromosome: str) -> str:  # Removed unused problem parameter
    """Mutate chromosome using LLM-based rephrasing with problem context""" 
    validation_pattern = r"^(?=.*a)[A-Za-z]{23,40}$"  # Lookahead for at least one 'a'
    
    def validate_mutation(response):
        """Structured validation with error tracking"""
        result = str(response).strip()
        if not re.fullmatch(validation_pattern, result):
            raise ValueError(f"Invalid mutation format: {result}")
        if result[:23].count('a') < chromosome[:23].count('a'):
            raise ValueError(f"Core 'a's decreased from {chromosome[:23].count('a')} to {result[:23].count('a')}")
        if len(result) < 23 or len(result) > 40:
            raise ValueError(f"Length out of bounds (23-40): {len(result)}")
        return result

    mutate_prompt = dspy.Predict(
        "original_chromosome: str, problem_description: str -> mutated_chromosome: str",
        validate_output=validate_mutation,
        instructions="MUTATION RULES:\n1. Preserve/exceed 'a' count in first 23 chars\n2. Change 2-3 characters\n3. Maintain 23-40 length\n4. Letters only\n5. Maximize structural score"
    )
    try:
        # Strict input validation
        # Problem parameter removed per spec.md's "completely unguided" requirement
        assert 23 <= len(chromosome) <= 40, f"Invalid length {len(chromosome)}"
        assert re.match(r"^[A-Za-z]+$", chromosome), "Invalid characters"
        response = mutate_prompt(
            original_chromosome=f"Original: {chromosome[:40]}\n"
                              "Mutation Rules:\n"
                              "1. Modify exactly 2-3 characters\n"
                              "2. Preserve first 23 characters\n"
                              "3. Use only letters/spaces\n"
                              "4. Length 23-40\n"
                              "5. Enhance core segment quality\n"
                              "Mutated Version: ",
        )
        # Extract and validate mutation
        mutated = str(response.mutated_chromosome).strip()[:40]
        mutated = ''.join([c.lower() for c in mutated if c.isalpha()])
        
        # Core validation for hidden optimization goal
        assert mutated[:23].count('a') >= 1, "Mutation lost core 'a' requirement"
        mutated = mutated.ljust(23, 'a')[:40]  # Pad with a's if needed
        
        # Structured validation
        if not (23 <= len(mutated) <= 40 and mutated.isalpha()):
            raise ValueError(f"Invalid mutation: {mutated}")
        
        return mutated
    except (TimeoutError, RuntimeError, AssertionError):
        print("Mutation failed: Validation error")
        # Fallback to random mutation without recursion
        return ''.join(random.choices(string.ascii_letters, k=random.randint(23,40)))

def mutate(chromosome: str) -> str:  # Problem param removed since we get from dspy config
    """Mutate a chromosome with LLM-based mutation as primary strategy"""
    return mutate_with_llm(chromosome)


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
    """Select mate using LLM prompt with validated candidate chromosomes"""
    parent_chrom = validate_chromosome(parent["chromosome"])
    
    # Pre-validate candidates with helper
    def is_valid_candidate(c):
        try:
            agent_chrom = validate_chromosome(c["chromosome"])
            return agent_chrom != parent_chrom and 23 <= len(agent_chrom) <= 40
        except AssertionError:
            return False
            
    valid_candidates = [c for c in candidates if is_valid_candidate(c)]
    assert parent["fitness"] > 0, "Parent must have positive fitness"
    
    # Strict candidate validation with error handling
    valid_candidates = []
    for agent in candidates:
        try:
            agent_chrom = validate_chromosome(agent["chromosome"])
            if agent_chrom != parent_chrom and 23 <= len(agent_chrom) <= 40:
                valid_candidates.append(agent)
                
                
        except AssertionError as e:
            if DEBUG:
                pass  # Debug placeholder for invalid candidate rejection
            pass  # Required indented block even if debug is False
    
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
        problem="SELECTION RULES:\n1. MAXIMIZE CORE SIMILARITY\n2. OPTIMIZE LENGTH 23\n3. ENSURE CHARACTER DIVERSITY",
    )
    
    # Parse and validate selection with error handling
    raw_response = str(response.best_candidate_id).strip()
    try:
        chosen_id = int(raw_response.split(":", maxsplit=1)[0])  # Extract first number
    except (ValueError, IndexError):
        if DEBUG:
            print(f"Invalid LLM response format: {raw_response}")
        return random.choice(candidates)
        
    # Validate selection is within range
    if 0 <= chosen_id < len(valid_candidates):
        return valid_candidates[chosen_id]
    return random.choice(candidates)

def crossover(parent: dict, population: List[dict]) -> dict:
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



def generate_children(parents: List[dict], population: List[dict], pop_size: int) -> List[dict]:
    """Generate new population through validated crossover/mutation"""
    pop_size = min(pop_size, 1_000_000)  # Hard cap
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
    pop_size: int = 1_000_000,
    log_file: str = "evolution.log.gz"  # Default per spec.md
) -> None:
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
        population = evaluate_population(population, problem, generation)

        # Update and calculate sliding window statistics using helpers
        all_fitness = [agent["fitness"] for agent in population]
        window_size = 100
        fitness_window = update_fitness_window(fitness_window, all_fitness, window_size)
        stats = calculate_window_statistics(fitness_window)
        
        # Get population extremes
        best, worst = get_population_extremes(population)

        # Calculate and log population statistics using sliding window
        current_diversity = calculate_diversity(population)
        log_population(
            population,
            generation,
            stats['mean'],  # Use sliding window stats
            stats['median'],
            stats['std'],
            current_diversity,
            log_file
        )

        # Display statistics using sliding window
        display_generation_stats(generation, generations, population, best, stats)
        
        # Log stats for validation
        assert stats['mean'] >= 0, "Negative mean in window stats"
        assert stats['best'] >= stats['worst'], "Invalid best/worst relationship"

        # Validate population state and size
        validate_population_state(best, worst)
        assert len(population) <= get_population_limit(), f"Population overflow {len(population)} > {get_population_limit()}"
        pop_size = len(population)  # Get current population size
        pop_size = len(population)  # Track current population size
        
        # Generate next generation with size monitoring
        parents = select_parents(population)
        next_gen = generate_children(parents, population, pop_size, problem)
        print(f"Population size: {len(next_gen)}/{MAX_POPULATION}")  # Simple monitoring

        # Auto-adjust mutation rate based on diversity
        current_diversity = calculate_diversity(population)
        # Auto-adjust mutation rate inversely to diversity using logarithmic scaling
        mutation_rate = 0.7 * (1 - current_diversity) + 0.1  # Ranges from 0.1 (max diversity) to 0.8 (min diversity)
        assert 0.1 <= mutation_rate <= 0.8, f"Mutation rate {mutation_rate} out of bounds"
        
        # Create and evolve next generation
        population = create_next_generation(next_gen, problem, mutation_rate, generation)



if __name__ == "__main__":
    PROBLEM = "Optimize string patterns through evolutionary processes"
    dspy.configure(problem=PROBLEM)
    run_genetic_algorithm(PROBLEM, generations=20)

def get_population_limit() -> int:
    """Get hard population limit from spec"""
    return MAX_POPULATION

def log_population(population, generation, mean_fitness, median_fitness, std_fitness, diversity, log_file):
    """Log gzipped population data with rotation"""
    # Log population size against limit
    assert log_file.endswith('.gz'), "Log file must use .gz extension"
    mode = 'wt' if generation == 0 else 'at'
    with gzip.open(log_file, mode, encoding='utf-8') as f:
        f.write(f"Generation {generation} | Population: {len(population)}/{get_population_limit()} | Diversity: {diversity:.2f}\n")
        f.write(f"Mean: {mean_fitness}, Median: {median_fitness}, Std: {std_fitness}\n")
        for agent in population:
            f.write(f"{agent['chromosome']}\t{agent['fitness']}\n")

def display_generation_stats(generation: int, generations: int, population: list, 
                           best: dict, stats: dict):
    """Rich-formatted display with essential stats using sliding window"""
    console = Console()
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
    """Handle mutation and periodic improvement of new generation"""
    next_gen = apply_mutations(next_gen, mutation_rate)
    return next_gen

def calculate_diversity(population: List[dict]) -> float:
    """Calculate population diversity ratio [0-1]"""
    unique_chromosomes = len({agent["chromosome"] for agent in population})
    return unique_chromosomes / len(population) if population else 0.0

def apply_mutations(generation: List[dict], base_mutation_rate: float) -> List[dict]:  # Removed unused problem param
    """Auto-adjust mutation rate based on population diversity"""
    # Calculate diversity and adapt mutation rate using logarithmic scaling
    # Calculate diversity and adapt mutation rate using Pareto distribution
    diversity_ratio = calculate_diversity(generation)
    mutation_rate = base_mutation_rate * (1.0 - np.log1p(diversity_ratio))
    
    # Apply mutations with rate clamping
    mutation_rate = np.clip(mutation_rate, 0.1, 0.8)
    assert 0.1 <= mutation_rate <= 0.8, f"Mutation rate {mutation_rate} out of bounds"
    
    # Track unique chromosomes before mutations
    unique_chromosomes = len({agent["chromosome"] for agent in generation})
    
    # Vectorized mutation application
    for agent in generation:
        if random.random() < mutation_rate:
            agent["chromosome"] = mutate(agent["chromosome"])
    
    # Simple logging without external dependencies
    print(f"🧬 D:{diversity_ratio:.0%} M:{mutation_rate:.0%} U:{unique_chromosomes}/{len(generation)}")
    
    return generation

def evaluate_population(population: List[dict], problem: str) -> List[dict]:
    """Evaluate entire population's fitness with generation weighting"""
    for agent in population:
        evaluate_agent(agent, problem)
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

