import dspy
import random
import string
from typing import List

# Configure DSPy with OpenRouter
lm = dspy.LM('openrouter/google/gemini-2.0-flash-001')
dspy.configure(lm=lm)

def create_agent(chromosome: str) -> dict:
    """Create a new agent as a dictionary"""
    return {
        'chromosome': chromosome,
        'fitness': 0.0
    }

def evaluate_agent(agent: dict, problem_description: str) -> float:
    """Evaluate the agent based on the optimization target"""
    # Limit chromosome length to 40 characters
    chromosome = agent['chromosome'][:40]
    
    # Calculate fitness based on the target rules
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
    import random
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
    population = initialize_population(pop_size)
    
    for generation in range(generations):
        # Evaluate all agents
        for agent in population:
            evaluate_agent(agent, problem)
        
        # Print current generation
        print(f"\nGeneration {generation + 1}")
        sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
        for agent in sorted_pop:
            print(f"Chromosome: {agent['chromosome'][:50]}... | Fitness: {agent['fitness']}")
        
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
        
        # Use LLM to improve top candidates
        for i in range(min(2, len(next_gen))):
            prompt = f"Improve this solution: {next_gen[i]['chromosome']}\nThe goal is: {problem}"
            response = lm(prompt)
            next_gen[i]['chromosome'] = response[:40]  # Limit to 40 chars
        
        population = next_gen

if __name__ == "__main__":
    PROBLEM = "Generate a string with many 'a's in first 23 chars and short after"
    run_genetic_algorithm(PROBLEM, generations=20, pop_size=10)
