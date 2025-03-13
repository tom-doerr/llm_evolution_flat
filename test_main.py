import dspy

class EvolutionOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_metric = dspy.Predict("chromosome -> mutated_chromosome")
        
    def optimize(self, population, metric):
        """Core optimization loop matching DSPy conventions"""
        best = max(population, key=lambda x: x["fitness"])
        return [create_agent(self.generate_metric(chromosome=best["chromosome"]).mutated_chromosome)]
