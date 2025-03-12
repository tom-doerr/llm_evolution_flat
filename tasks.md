# Genetic Algorithm Task List (Prioritized)

- Create concise console output with Rich formatting  
- Add sliding window for last 100 evaluation stats  
- Tune Pareto distribution parameters  
- Implement chromosome deduplication checks  

## Recently Completed
[x] Implement LLM-based mutation per spec  
[x] Add weighted parent selection using Pareto distribution  

## Recently Completed
[x] Add population statistics (mean/median/std)  
[x] Harden LM response validation  
[x] Implement mutation rate configuration  
[x] Add boundary condition assertions  
[x] Improve LM timeout handling with retries  
[x] Implement vectorized population initialization  
[x] Add compressed binary logging  
[x] Implement LLM-based mutation framework  

## Current Issues/Notes
- LLM-based mutation not yet implemented per spec
- Parent selection uses simple truncation instead of Pareto distribution
- Statistics show full history instead of last 100 evals
- Chromosome truncation during creation needs logging
- Fitness calculation needs penalty system for oversize chromosomes

## Next Steps
1. Add boundary condition tests (empty strings, min/max lengths)
2. Implement property-based tests for genetic operations
3. Add mutation rate configuration
