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


def select_parents(population: List[dict], fitness_window: list) -> List[dict]:
    """Select parents using sliding window of fitness^2 weighted sampling"""
    window = fitness_window[-WINDOW_SIZE:]
    candidates = [a for a in population if a['fitness'] in window]
    
    # Pareto distribution weighting by fitness^2 per spec.md
    fitness_scores = np.array([a['fitness']**2 + 1e-6 for a in candidates], dtype=np.float64)
    pareto_weights = np.random.pareto(fitness_scores, size=len(fitness_scores))
    return [candidates[i] for i in np.random.default_rng().choice(
        len(candidates), 
        size=min(len(candidates)//2, MAX_POPULATION),
        p=pareto_weights/np.sum(pareto_weights),
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
        chromosome=[agent["chromosome"]]*3,
        instructions=[agent.get("mutation_chromosome", "Change 1-2 chars post-23")]*3,
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
        parent_chromosome=parent["chromosome"],
        candidate_chromosomes=[c["chromosome"] for c in valid],
        temperature=0.7,
        top_p=0.9
    )

    selected = str(response.selected_mate).strip()[:40]
    for c in valid:
        if c["chromosome"] == selected:
            return c

    weights = np.array([c["fitness"]**2 for c in valid], dtype=np.float64)
    return valid[np.random.choice(len(valid), p=weights/weights.sum())]

def crossover(parent: dict, population: List[dict]) -> dict:
    """Create child through LLM-assisted mate selection"""
    candidates = random.choices(
        population=population[-WINDOW_SIZE:],
        weights=np.array([a["fitness"]**2 + 1e-6 for a in population[-WINDOW_SIZE:]], dtype=np.float64),
        k=min(5, len(population))
    )
    
    mate = llm_select_mate(parent, candidates)
    split_point = random.randint(1, len(parent["chromosome"])-1)
    
    try:
        new_chromosome = parent["chromosome"][:split_point] + mate["chromosome"][split_point:]
        validate_chromosome(new_chromosome)
        if new_chromosome[:23].count('a') < parent["chromosome"][:23].count('a'):
            raise ValueError("Core 'a' count decreased")
        return create_agent(new_chromosome)
    except (AssertionError, ValueError):
        return create_agent(mutate(parent["chromosome"]))



def generate_children(parents: List[dict], population: List[dict]) -> List[dict]:
    """Generate new population through validated crossover/mutation"""
    next_gen = parents.copy()
    target_size = min(len(population), MAX_POPULATION)
    
    for _ in range(min(MAX_POPULATION, len(parents)*2)):
        parent = random.choice(parents)
        try:
            next_gen.append(crossover(parent, population))
        except ValueError:
            next_gen.append(create_agent(mutate(parent["chromosome"])))
    
    return next_gen[:MAX_POPULATION]

MAX_POPULATION = 1_000_000  # Hard cap from spec

def get_population_extremes(population: List[dict]) -> tuple:
    """Get best and worst agents from population"""
    sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
    return sorted_pop[0], sorted_pop[-1]

def run_genetic_algorithm(generations: int = 10, pop_size: int = 1_000_000) -> None:
    """Run genetic algorithm with optimized logging and scaling"""
    pop_size = min(pop_size, MAX_POPULATION)
    assert 1 < pop_size <= MAX_POPULATION, f"Population size must be 2-{MAX_POPULATION}"
    assert generations > 0, "Generations must be positive"

    population = initialize_population(pop_size)
    fitness_window = []

    open("evolution.log", "w").close()  # Empty log file per spec.md

    for generation in range(generations):
        population = evaluate_population(population)
        fitness_window = update_fitness_window(fitness_window, [a["fitness"] for a in population])
        stats = calculate_window_statistics(fitness_window)
        
        log_population(population, generation, stats)
        display_generation_stats(generation, generations, population, stats)
        
        # Trim population with weighted sampling
        population = sorted(population, key=lambda x: -x['fitness'])
        fitness_weights = np.array([a['fitness']**2 + 1e-6 for a in population], dtype=np.float64)
        selected_indices = np.random.choice(
            len(population),
            size=min(len(population), MAX_POPULATION),
            p=fitness_weights/fitness_weights.sum()
        )
        population = [population[i] for i in selected_indices]
        
        parents = select_parents(population, fitness_window)
        population = create_next_generation(generate_children(parents, population), 0.25)



if __name__ == "__main__":
    PROBLEM = "Optimize string patterns through evolutionary processes"
    dspy.configure(problem=PROBLEM)
    run_genetic_algorithm(generations=20)

def log_population(population: List[dict], generation: int, stats: dict) -> None:
    """Log population data with rotation"""
    with open("evolution.log", "a" if generation else "w", encoding='utf-8') as f:
        best = max(population, key=lambda x: x['fitness'])
        worst = min(population, key=lambda x: x['fitness'])
        f.write(f"Generation {generation} | Size: {len(population)} | "
                f"Mean: {stats['mean']:.2f} | Best: {best['fitness']:.2f} | "
                f"Worst: {worst['fitness']:.2f}\n")

def display_generation_stats(generation: int, generations: int, population: List[dict], stats: dict):
    """Rich-formatted display with essential stats"""
    diversity = calculate_diversity(population)
    Console().print(Panel(
        f"[bold]Generation {generation}/{generations}[/]\n"
        f"📊 Mean: {stats['mean']:.2f} | 📈 Best: {stats['best']:.2f}\n"
        f"🌐 Diversity: {diversity:.1%} | 👥 Size: {len(population)}",
        title="Evolution Progress",
        style="blue"
    ))



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


def get_population_limit() -> int:
    """Get hard population limit from spec"""
    return MAX_POPULATION

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

