import argparse
import pytest
from unittest.mock import patch, MagicMock
import main 
import dspy

@pytest.fixture
def mock_lm():
    lm = dspy.LM("openrouter/mock_model")  # Use valid openrouter prefix per spec
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
    mock_lm.model = "openrouter/mock_model"  # Valid openrouter prefix per spec
    
    agent = main.create_agent("test")
    # Create proper argparse namespace with required fields
    args = argparse.Namespace(verbose=False, threads=1)
    mutated_chromo = main.mutate(agent, args)
    assert mutated_chromo != agent["chromosome"]
    mutated_agent = main.create_agent(mutated_chromo)
    assert len(mutated_agent["chromosome"]) >= 23

def test_crossover_no_duplicates():
    parent1 = main.create_agent("a"*23)
    parent2 = main.create_agent("b"*23)
    child = main.crossover(parent1, [parent2])
    assert 23 <= len(child["chromosome"]) <= 40
    assert child["chromosome"] != parent1["chromosome"]
    assert child["chromosome"] != parent2["chromosome"]
    assert main.validate_chromosome(child["chromosome"])

def test_trim_population_deduplication():
    pop = [main.create_agent("aaa"), main.create_agent("aaa"), main.create_agent("bbb")]
    trimmed = main.trim_population(pop, 2)
    assert len(trimmed) == 2
    assert len({a["chromosome"] for a in trimmed}) == 2
class EvolutionOptimizer(dspy.Module):
    """DSPy optimizer implementing evolutionary strategies"""
    def __init__(self, population_size=1000):
        super().__init__()
        self.population_size = population_size
        self.generate_metric = dspy.Predict("chromosome -> mutated_chromosome")
        
    def forward(self, population):
        """Run one evolution step"""
        parents = select_parents(population)
        children = generate_children(parents, population, argparse.Namespace(verbose=False, threads=1))
        return trim_population(parents + children, self.population_size)
