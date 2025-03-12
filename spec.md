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


as a task i want to optimize this hidden goal for testing: reward increases for every a for the first 23 characters and decreases for every character after 23 characters. limit token output to 40 for the dspy lm


when you implemented all features, improve test coverage
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




