import os
import openai
from typing import List, Dict, Any

# Configure OpenRouter
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENROUTER_API_KEY")

class GeneticAgent:
    def __init__(self, chromosome: str):
        self.chromosome = chromosome
        self.fitness = 0.0

    def evaluate(self, problem_description: str) -> float:
        """Evaluate the agent by making an LLM request"""
        try:
            response = openai.ChatCompletion.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {"role": "system", "content": problem_description},
                    {"role": "user", "content": self.chromosome}
                ]
            )
            # For now, just use response length as fitness
            self.fitness = len(response.choices[0].message.content)
            return self.fitness
        except Exception as e:
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
    problem = "Optimize this solution for maximum efficiency"
    run_genetic_algorithm(problem)
