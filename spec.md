chromosomes to separate functionality to make sure each agent has info for every area
as much evolution as possible, even for combination and mate selection
easy way to run it on problems
would be great if this would work as a dspy optimizer
easy to understand output when running the program
display population size
rewards can be negative
there should be little, but information dense output so we don't fill up context
the program should log more detailed information to a file
the log file should be emptied when we start the program
keep it simple
parent selection should use:
- Pareto distribution weighting by fitness^2 
- Weighted sampling without replacement
include mean, median, std deviation for population reward
set the default population size limit to one million
mutation should be llm based, specifically requesting to:
mating should work by loading the dna of agents into the prompt and then using the mating chromosome/prompt of the agent allowed to mate to select one of the candidates
the candidates list should be created by weighted sampling based on score without replacement
statistics should:
- Use sliding window of last 100 evaluations
- Show mean, median, std deviation
- Track best/worst in current population


as a task i want to optimize this hidden goal for testing: reward increases for every a for the first 23 characters and decreases for every character after 23 characters. limit token output to 40 for the dspy lm
don't do reward shaping
this is supposed to be hard, I don't expect good results
don't reveal the goal to the optimization process
don't explicitely write anywhere that we are maximizing 'a's, only implicit through reward


keep all of it in a single file
keep it low complexity
use this model: openrouter/google/gemini-2.0-flash-001
when I type 'c', i mean continue working on implementing spec.md
don't use OOP if possible
use pure functions where possible
in DSPy, you can do lm = dspy.LM('openrouter/google/gemini-2.0-flash-001') to load a model
we might want to use rich for output
don't add fallbacks, fix the real issue
use many asserts with meaningful messages
the task list should be sorted by priority
update the task list frequently
remove duplicates from the task list 
instead of creating unit tests, lets just use many assertions 

don't edit spec.md, follow what is in it 
don't use docstrings, use comments when helpful to explain the why
don't use caching for the llm requests
don't work on error handling, the code doesn't need to be reliable right now
don't use a problem description at all, it should be completely unguided for now
don't set an initial prompt but instead use just an empty string 

do use TODO comments a lot
do many edits at once if possible 
do fix the issues in issues.txt if there are any
