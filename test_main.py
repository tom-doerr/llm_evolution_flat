import dspy

class EvolutionOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_metric = dspy.Predict("chromosome -> mutated_chromosome")
        
    def optimize(self, population, metric):
        """Core optimization loop matching DSPy conventions"""
        best = max(population, key=lambda x: x["fitness"])
        return [create_agent(self.generate_metric(chromosome=best["chromosome"]).mutated_chromosome)]
import pytest
from unittest.mock import patch, MagicMock
import main
import dspy

@pytest.fixture
def mock_lm():
    lm = dspy.LM("mock_model")
    lm.return_value = MagicMock()
    return lm

def test_initial_population():
    pop = main.initialize_population(10)
    assert len(pop) == 10
    for agent in pop:
        assert len(agent["chromosome"]) >= 23
        assert len(agent["chromosome"]) <= 40

def test_mutation_mock(mock_lm):
    main.dspy.settings.lm = mock_lm
    mock_lm.return_value = MagicMock()
    mock_lm.return_value.completions = ["aaaaaabbbccc"]
    
    agent = main.create_agent("test")
    mutated = main.mutate(agent, argparse.Namespace(verbose=False))
    assert mutated["chromosome"] != "test"

def test_crossover_no_duplicates():
    parent1 = main.create_agent("a"*23)
    parent2 = main.create_agent("b"*23)
    child = main.crossover(parent1, [parent2])
    assert child["chromosome"] != parent1["chromosome"]
    assert child["chromosome"] != parent2["chromosome"]

def test_trim_population_deduplication():
    pop = [main.create_agent("aaa"), main.create_agent("aaa"), main.create_agent("bbb")]
    trimmed = main.trim_population(pop, 2)
    assert len(trimmed) == 2
    assert len({a["chromosome"] for a in trimmed}) == 2
