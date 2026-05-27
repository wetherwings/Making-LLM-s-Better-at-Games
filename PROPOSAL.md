# Proposal

## Capability
What is the scientific capability? What makes it interesting and challenging?

_Scientific Task:_ Forecasting optimal actions in non-cooperative sequential move games. 

_What makes it interesting:_ Forecasting optimal actions in non-cooperative sequential-move games challenges current large language models by demanding they execute rigorous backward induction and multi-step lookahead across extensive-form decision trees. 

## Evidence of Frontier Failure
~5 concrete examples where a frontier model (Gemini, Claude, ...) gets the task wrong or performs poorly. Include the prompts and outputs.
Prompt used: 

```text
Role: You are a competitive player in the game of Chopsticks. Your objective is to win by making both of my hands reach exactly 0 fingers. You must play with the intent to defeat me.

The Game Mechanics:

The State: Each player has a Left (L) and Right (R) hand. We start at [1, 1] vs [1, 1].

The Goal: A hand is "out" (0) when it reaches exactly 5 fingers. If a hit results in a sum greater than 5, the hand remains at the remainder (e.g., $3 + 3 = 6$, which becomes 1).

Actions:

Attack: Use one of your "live" hands to hit one of mine. Add your fingers to mine.

Split: You may redistribute your total fingers between your two hands (e.g., if you have [4, 0], you can split to [2, 2]). You cannot "swap" (e.g., [3, 1] to [1, 3] is illegal).

Winning: You win if both of my hands are 0. I win if both of yours are 0.


```
Claude 4.6 Sonnet 
![](https://github.com/bu-ds595/final-project-localminimaclub-1/blob/proposal/example%20outputs/sonnet-4.6-1.png)
![](https://github.com/bu-ds595/final-project-localminimaclub-1/blob/proposal/example%20outputs/sonnet-4.6-2.png)
Gemini 2.5 Pro
![](https://github.com/bu-ds595/final-project-localminimaclub-1/blob/proposal/example%20outputs/gemini-2.5-pro-1.png)
Gemini 3 Flash
![](https://github.com/bu-ds595/final-project-localminimaclub-1/blob/proposal/example%20outputs/gemini-3-flash-1.png)
GPT-5.3
![](https://github.com/bu-ds595/final-project-localminimaclub-1/blob/proposal/example%20outputs/GPT-5.3-1.png)

## Eval Plan
How will you measure success? What's the input format, output format, scaffolding (e.g., tool use), and scoring method?

We could find basic other systems that are able to compete and win at these games and calculate the percentage of winning that the LLM does which gives less consistent results and thus requires more steps done to test. Alternatively, with some of the complete games where every option is known, we could calculate the amount of possible wins versus losses (or draws) still attainable based on an action made by the LLM and reward based on picking the best sequence and its expected result. Thus regret/loss would be subtracting the option it took by the possible best option it could have taken for the state the game was in at that time which we would want minimally to be at zero. This would be reversed if using rewards instead of regrets where it would need to be given some reward for choosing an option close to (or precisely) the best option. 

The input can either come from a separate AI capable at the game at varying levels if not a complete game. Or an algorithm with full knowledge picking its best case scenario (its most average wins to create the best competitor). It will be formatted in ASCII or list format dependent on the game as seen in evidence above.

Output should be in the same format and then converted to be sent to the algorithm sending the next input. If the LLM does not give the correct format to convert it back we would have to ask it to do so again most likely unless we are training manually.

## Data Plan
Where will your training data come from? Rough estimate of size. Do you plan to do SFT, RL, scaffolding improvements etc?

Synthetic data can be generated with relative ease by running simulators of sample games at length. Those with a reasonably searchable result space (tic-tac-toe, chopsticks, etc…) can be mapped out entirely by playing every possible game which are the best initial options for our testing. We can use both SFT and RL. We can have SFT used as an initial guide on how to figure out the best options for the various games by giving some guidance on what should or should not be picked. SFT might not generalize as well for the purpose of improving backward induction or predictive reasoning though it is likely to improve specific results. We can have an RL kind of algorithm that focuses on minimizing loss based on what is the most optimal option at any time for those that can be mapped out entirely. This can be used to evaluate if the LLM is doing well or not. At this point it would have to have some tool or algorithm created to calculate either a loss based on the best possible option or an opponent who will play against the LLM.

