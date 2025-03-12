# Genetic Algorithm Task List (Prioritized)

- Create concise console output with Rich formatting  
- Implement LLM-based mutation per spec  
- Add weighted parent selection using Pareto distribution  
- Add sliding window for last 100 evaluation stats  

## Recently Completed
[x] Add population statistics (mean/median/std)  
[x] Harden LM response validation  
[x] Implement mutation rate configuration  
[x] Add boundary condition assertions  

## Recently Completed
[x] Add population statistics (mean/median/std)  
[x] Harden LM response validation  
[x] Implement mutation rate configuration  
[x] Add boundary condition assertions  
[x] Improve LM timeout handling with retries  
[x] Implement vectorized population initialization  
[x] Add compressed binary logging  

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
