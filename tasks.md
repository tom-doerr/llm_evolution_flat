# Genetic Algorithm Task List (Prioritized)

1. Implement basic genetic algorithm structure with fitness evaluation
2. Add DSPy integration for LLM-assisted optimization
3. Set up logging system with detailed evolution history
4. Create concise console output with Rich formatting
5. Add test coverage for core functions
6. Implement chromosome validation system
7. Add negative reward handling in fitness calculation
8. Create chromosome length limit (40 chars) enforcement
9. Develop property-based tests for genetic operations
10. Implement file logging cleanup on startup

## Current Issues/Notes (notes.md draft)
- Need to harden LM response validation
- Fitness calculation needs penalty system for oversize chromosomes
- Should add input validation for all public functions
- Consider adding crossover rate parameter
- Need better error handling for LM timeouts

## Next Steps
1. Implement fitness function with penalty system
2. Add input validation guards
3. Create test cases for boundary conditions
