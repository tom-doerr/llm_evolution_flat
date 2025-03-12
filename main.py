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
            # Create a proper prompt
            prompt = f"Problem: {problem_description}\nSolution: {self.chromosome}\n\nEvaluate this solution's quality on a scale from 0 to 10, where 10 is perfect:"
            
            # Get LLM response
            response = lm(prompt)
            
            # Try to extract a numeric score from the response
            try:
                # Look for a number between 0 and 10 in the response
                self.fitness = float(next((word for word in response.split() if word.replace('.', '').isdigit()), 1.0)
                # Ensure fitness is between 0 and 10
                self.fitness = max(0.0, min(10.0, self.fitness))
                return self.fitness
            except (ValueError, StopIteration):
                # If we can't parse a number, use default fitness
                return 1.0
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
