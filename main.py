import dspy
from typing import List

# Configure DSPy with OpenRouter
lm = dspy.LM('openrouter/google/gemini-2.0-flash-001')
dspy.configure(lm=lm)

class GeneticAgent:
    def __init__(self, chromosome: str):
        self.chromosome = chromosome
        self.fitness = 0.0

    def evaluate(self, problem_description: str) -> float:
        """Evaluate the agent based on the optimization target"""
        # Limit chromosome length to 40 characters
        chromosome = self.chromosome[:40]
        
        # Calculate fitness based on the target rules
        fitness = 0.0
        
        # First 23 characters: +1 for each 'a'
        first_part = chromosome[:23]
        fitness += first_part.lower().count('a')
        
        # Remaining characters: -1 for each character
        remaining = chromosome[23:]
        fitness -= len(remaining)
        
        # Ensure fitness is not negative
        self.fitness = max(0.0, fitness)
        return self.fitness

def initialize_population(pop_size: int) -> List[GeneticAgent]:
    """Create initial population with random chromosomes"""
    import random
    import string
    
    def random_chromosome():
        # Generate random string with letters and spaces
        length = random.randint(20, 40)
        return ''.join(random.choices(string.ascii_letters + ' ', k=length))
    
    return [GeneticAgent(chromosome=random_chromosome()) for _ in range(pop_size)]

def run_genetic_algorithm(problem: str, generations: int = 10, pop_size: int = 5):
    """Run basic genetic algorithm"""
    population = initialize_population(pop_size)
    
    for generation in range(generations):
        print(f"\nGeneration {generation + 1}")
        for agent in population:
            fitness = agent.evaluate(problem)
            print(f"Chromosome: {agent.chromosome[:50]}... | Fitness: {fitness}")

if __name__ == "__main__":
    PROBLEM = "Generate a string with many 'a's in first 23 chars and short after"
    run_genetic_algorithm(PROBLEM, generations=20, pop_size=10)
