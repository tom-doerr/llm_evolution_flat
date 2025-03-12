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
        """Evaluate the agent by making an LLM request"""
        try:
            response = lm(self.chromosome)
            # For now, just use response length as fitness
            self.fitness = len(response)
            return self.fitness
        except RuntimeError as e:
            print(f"Error in LLM request: {e}")
            return 0.0

def initialize_population(pop_size: int) -> List[GeneticAgent]:
    """Create initial population with random chromosomes"""
    return [GeneticAgent(chromosome=f"Initial chromosome {i}") for i in range(pop_size)]

def run_genetic_algorithm(problem: str, generations: int = 10, pop_size: int = 5):
    """Run basic genetic algorithm"""
    population = initialize_population(pop_size)
    
    for generation in range(generations):
        print(f"\nGeneration {generation + 1}")
        for agent in population:
            fitness = agent.evaluate(problem)
            print(f"Chromosome: {agent.chromosome[:50]}... | Fitness: {fitness}")

if __name__ == "__main__":
    PROBLEM = "Optimize this solution for maximum efficiency"
    run_genetic_algorithm(PROBLEM)
