import random
import string
from typing import List
import dspy
import logging

# Configure logging
logging.basicConfig(
    filename='evolution.log',
    filemode='w',  # Empty file on start
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Configure DSPy with OpenRouter and timeout
lm = dspy.LM('openrouter/google/gemini-2.0-flash-001', max_tokens=40, timeout=10)
dspy.configure(lm=lm)

# Validate configuration
assert isinstance(lm, dspy.LM), "LM configuration failed"

def create_agent(chromosome: str) -> dict:
    """Create a new agent as a dictionary"""
    # Ensure chromosome is a non-empty string
    if isinstance(chromosome, list):
        chromosome = ''.join(chromosome)
    chromosome = str(chromosome).strip()
    if not chromosome:
        # Fallback to random chromosome if empty
        length = random.randint(20, 40)
        chromosome = ''.join(random.choices(string.ascii_letters + ' ', k=length))
    return {
        'chromosome': chromosome,
        'fitness': 0.0
    }

def evaluate_agent(agent: dict, _problem_description: str) -> float:
    """Evaluate the agent based on the optimization target"""
    # Ensure chromosome is a string
    chromosome = str(agent['chromosome'])
    
    # Calculate fitness based on the strict target rules
    fitness = 0.0
    
    # First 23 characters: +1 for each 'a'
    first_part = chromosome[:23]
    fitness += first_part.lower().count('a')
    
    # Remaining characters: -1 for each character
    remaining = chromosome[23:]
    fitness -= len(remaining)
    
    # Ensure fitness is not negative
    agent['fitness'] = max(0.0, fitness)
    return agent['fitness']

def initialize_population(pop_size: int) -> List[dict]:
    """Create initial population with random chromosomes"""
    def random_chromosome():
        # Generate random string with letters and spaces
        length = random.randint(20, 40)
        return ''.join(random.choices(string.ascii_letters + ' ', k=length))
    
    return [create_agent(random_chromosome()) for _ in range(pop_size)]

def select_parents(population: List[dict]) -> List[dict]:
    """Select top 50% of population as parents"""
    sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
    return sorted_pop[:len(population)//2]

def mutate(chromosome: str) -> str:
    """Mutate a chromosome by replacing one random character"""
    idx = random.randint(0, len(chromosome)-1)
    new_char = random.choice(string.ascii_letters + ' ')
    return chromosome[:idx] + new_char + chromosome[idx+1:]

def crossover(parent1: dict, parent2: dict) -> dict:
    """Create child by combining parts of parent chromosomes"""
    split = random.randint(1, len(parent1['chromosome'])-1)
    new_chromosome = parent1['chromosome'][:split] + parent2['chromosome'][split:]
    return create_agent(new_chromosome)

def run_genetic_algorithm(problem: str, generations: int = 10, pop_size: int = 5):
    """Run genetic algorithm with LLM-assisted evolution"""
    assert pop_size > 1, "Population size must be greater than 1"
    assert generations > 0, "Number of generations must be positive"
    
    population = initialize_population(pop_size)
    logging.info(f"Starting evolution with population size {pop_size}")
    
    for generation in range(generations):
        # Evaluate all agents
        for agent in population:
            evaluate_agent(agent, problem)
        
        # Print and log current generation
        sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
        best = sorted_pop[0]
        worst = sorted_pop[-1]
        
        # Information-dense output
        print(f"\nGen {generation+1}/{generations} | Pop: {pop_size}")
        print(f"Best: {best['chromosome'][:23]}... (fit:{best['fitness']})")
        print(f"Worst: {worst['chromosome'][:23]}... (fit:{worst['fitness']})")
        
        # Log detailed info
        logging.info(f"Generation {generation+1}")
        logging.info(f"Best: {best['chromosome']} (fitness: {best['fitness']})")
        logging.info(f"Worst: {worst['chromosome']} (fitness: {worst['fitness']})")
        
        # Select parents and create next generation
        parents = select_parents(population)
        next_gen = parents.copy()
        
        # Create children through crossover
        while len(next_gen) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            next_gen.append(child)
        
        # Mutate some children
        for i in range(len(next_gen)//2):
            next_gen[i]['chromosome'] = mutate(next_gen[i]['chromosome'])
        
        # Use LLM to improve top candidates (only every 5 generations)
        if generation % 5 == 0:
            for i in range(min(2, len(next_gen))):
                prompt = f"Improve this solution: {next_gen[i]['chromosome']}\nThe goal is: {problem}"
                try:
                    response = lm(prompt)
                    # Ensure response is valid
                    if response and isinstance(response, str) and len(response) > 0:
                        next_gen[i]['chromosome'] = response
                    else:
                        # If invalid response, mutate instead
                        next_gen[i]['chromosome'] = mutate(next_gen[i]['chromosome'])
                except Exception as e:
                    print(f"LLM improvement failed: {str(e)}")
                    # If LLM fails, mutate instead
                    next_gen[i]['chromosome'] = mutate(next_gen[i]['chromosome'])
        
        population = next_gen

if __name__ == "__main__":
    PROBLEM = "Generate a string with many 'a's in first 23 chars and short after"
    run_genetic_algorithm(PROBLEM, generations=20, pop_size=10)
