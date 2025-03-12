chromosomes to seperate functionality to make sure each agent has info for every area
as much evolution as possible, even for combination and mate selection
easy way to run it on problems
would be great if this would work as a dspy optimizer
easy to understand output when running the program
display population size
rewards can be negative
there should be little, but information dense otput so we don't fill up context
the program should log more detailed information to a file
the log file should be emptied when we start the program
keep it simple
the better the reward, the higher the probability for mating, pareto distribution
include mean, median, std deviation for population reward
set the default population size limit to one million
mutation should be llm based, maybe by prompting the llm to rephrase the text
mating should work by loading the dna of agents into the prompt and then using the mating chromosome/prompt of the agent allowed to mate to select one of the candidates
the canidates list should be created by weighted sampling based on score without replacement


as a task i want to optimize this hidden goal for testing: reward increases for every a for the first 23 characters and decreases for every character after 23 characters. limit token output to 40 for the dspy lm
don't do reward shapping
this is supposed to be hard, I don't expect good results


keep notes in notes.md about what you learned, what current issues are and plans
keep all of it in a single file
keep it low complexity
start with simple versions first, i always want to have something to run, we add features as we work on it
use this model: openrouter/google/gemini-2.0-flash-001
with c i mean continue
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
don't enumerate the task list
don't use caching for the llm requests
don't work on error handling, the code doesn't need to be reliable right now
do use TODO comments a lot




