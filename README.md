# Localminimaclub-1: Making LLMs Better at Games
**Team:** Leonardo Martins, Jelle Hendriks, Athila Koli


## 1. Capability

### What scientific task are you targeting? Why do LLMs struggle at it?

**Forecasting optimal actions in sequential-move, perfect-information games.** We target *backward induction* — reasoning from terminal game states back to the current position to find the provably optimal move. This requires a model to:

1. Enumerate **all** legal moves from the current position
2. Evaluate each move by simulating opponent responses recursively
3. Select the move that leads to the best outcome under optimal play

We target three games: **Tic-Tac-Toe**, **Connect 4**, and **Chopsticks**.

Current LLMs rely on **local heuristics instead of genuine backward induction**. They pick moves that look good immediately rather than reasoning from the end of the game backwards. This is a clean, measurable failure — every decision can be scored with a ground-truth oracle.

**Five concrete frontier failures (full game rules provided in every prompt):**

| Model | Game | Failure |
|---|---|---|
| Claude Sonnet 4.6 | Chopsticks | Missed a forced win two moves ahead — no lookahead performed |
| Gemini 2.5 Pro | Chopsticks | Applied "asymmetry is good" heuristic without checking result — wasted turn |
| GPT-4o | Chopsticks | Did not enumerate split as a legal move — lost a won position |
| Claude Sonnet 4.6 | Tic-Tac-Toe | Missed fork winning in 2 moves — chose draw strategy when win was available |
| GPT-4o | Tic-Tac-Toe | Played offense while ignoring a forced loss next turn |

All five share the same root cause: **pattern matching instead of genuine lookahead.**

---

## 2. Evaluation

### SFT Benchmark — Chopsticks

- **Examples:** 1,152 total states (exhaustive). Train: 979 | Eval: **173 held-out examples** (well above the 50+ minimum, seed 42, never seen in training)
- **Oracle:** Exact minimax — every label is provably optimal
- **Input format:** Board state in natural language + all legal moves explicitly listed
- **Output format:** Model must end response with `MOVE: <action>`
- **Correctness:** Automated regex parser. Two metrics:
  - **Optimality Rate (OR):** `|{s : a_model(s) ∈ A*(s)}| / |S_eval|` — fraction of decisions matching oracle
  - **Expected Score Loss (ESL):** `avg[ V*(s) − V(a_model, s) ]` — average regret vs best move. ESL = 0 is perfect.
- One retry if unparseable, scored 0 on second failure.

### SFT Benchmark — Tic-Tac-Toe

- **Examples:** 5,478 total states (exhaustive). Train: 7,684 | Eval: 1,356 (held out, seed 42, never seen in training)
- **Oracle:** Exact minimax — every label is provably optimal
- **Input format:** Board state as ASCII grid (positions 0-8) + all legal moves listed
- **Output format:** Model must end response with MOVE: <position 0-8>
- **Correctness:** Same automated scoring as Chopsticks — OR and ESL vs minimax oracle
- **Note:** 7,684 train > 5,478 total states because each board position is presented with two prompt ---framings (standard + predict-opponent's-move)

### RL Benchmark — Connect 4

- **Examples:** 17,000 training positions | 3,000 eval positions
- **Eval setup:** Model plays as Player 2 against 3 bot skill levels (tests role generalization — trained as P1)
- **Bots:** Random, Somewhat Smart, Semi-MinMax (multi-move lookahead)
- **Input format:** Board state in natural language + legal columns listed
- **Output format:** `MOVE: <column 0-6>`
- **Correctness:** Win rate, Draw rate, Loss rate, Illegal move rate — all automated

### Running the Eval Scripts

```bash
pip install uv

uv venv

uv pip install tinker-cookbook numpy matplotlib seaborn tqdm asyncio

export TINKER_API_KEY="your-api-key-here"

#on Mac/Linux, put it as a environment variable in Windows

# Test basic functionality (no API key required)
python run_pipeline.py

# Test Tinker integration (API key required)
python test_tinker_integration.py

# Run the complete training and evaluation pipeline
python main_training_pipeline.py

```

All eval scripts are in the repo root and fully runnable.

---

## 3. Training

### Data

All training data generated synthetically using Python solvers — no human labeling required. All datasets are in `./Dataset/` as JSONL files, split 85/15 train/eval with fixed seed 42.

| Game | Generation Method | Train | Eval | Total | Location |
|---|---|---|---|---|---|
| Chopsticks | Exhaustive minimax (all 1,152 states) | 979 | 173 | 1,152 | `Dataset/chopsticks_*.jsonl` |
| Tic-Tac-Toe | Exhaustive minimax (all 5,478 states) | 7,684 | 1,356 | 9,040 | `Dataset/ttt_*.jsonl` |
| Connect 4 | Alpha-beta depth 2, 20k sampled positions | 17,000 | 3,000 | 20,000 | `Dataset/c4_*.jsonl` |

**How the datasets were generated:**

- **Chopsticks & Tic-Tac-Toe:** A Python minimax solver exhaustively solved every possible game state. For Chopsticks there are 5⁴ = 625 unique hand configurations (1,152 non-terminal states when accounting for both players' turns). For Tic-Tac-Toe there are 5,478 valid non-terminal board positions. Every label is provably optimal.

- **Connect 4:** Too large for exhaustive search (~4.5 trillion positions), so we sampled 20,000 diverse positions by simulating random games then scored each with alpha-beta search at depth 2. Labels are strong but not provably perfect.

**Dataset record format** (each `.jsonl` line):

```json
{
  "prompt": "You are Player 1 in Chopsticks. Hands: L=1, R=2. Opponent: L=0, R=4. Legal moves: [...]. Reason step by step, output MOVE: <move>",
  "completion": "Reasoning:\n- [OPT] attack L->oppR score -1\n- [sub] ...\nMOVE: attack L->oppR",
  "state": [1, 2, 0, 4],
  "turn": 1,
  "outcome": "P1_wins",
  "optimal_moves": ["P1.L(1)->P0.R=0"],
  "suboptimal_moves": ["P1.R(2)->P0.R=1"]
}
```

The `optimal_moves` field is used directly by the eval scripts to score model outputs. The `completion` field with chain-of-thought scoring of all moves is what the model trains on.

Each SFT example is a `(prompt, completion)` pair. The completion includes chain-of-thought that scores **all** legal moves before naming the optimal — this is the key design choice that teaches move enumeration rather than pattern matching:

```
Reasoning:
- My hands: [1, 2]  |  Opponent: [0, 4]
- Evaluating 4 legal moves:
  [✓ OPTIMAL] attack L(1)→oppR(4)=0  score -1 (P1 wins)
  [  subopt ] attack R(2)→oppR(4)=1  score  0 (draw)
  [  subopt ] split [1,2]→[0,3]      score  0 (draw)
  [  subopt ] split [1,2]→[3,0]      score  0 (draw)
MOVE: attack L→oppR
```

### Model and Method

**Base model:** `meta-llama/Llama-3.1-8B-Instruct`  
**Fine-tuning:** LoRA via Tinker API (rank 32, LR 1e-4, batch size 8)

| Game | Method | Steps | Cost |
|---|---|---|---|
| Chopsticks | SFT | 244 batches (2 epochs × 122) | ~$0.50 |
| Tic-Tac-Toe | SFT | ~1,920 batches (2 epochs × 960) | ~$3.00 |
| Connect 4 | SFT + RL (GRPO) | SFT pre-train + 100 RL episodes | ~$23.00 |

**RL setup (Connect 4 GRPO):** Model trains as Player 1 against bots of increasing skill (Random → Smart → Semi-MinMax). Reward: +1 win, +0.5 draw, 0 ongoing, −1 loss or illegal move.

**Chopsticks training loss curve:**

```
Epoch 1: 695.2 → 138.0 → 59.6 → 32.1 → 12.1
Epoch 2: 28.2 → 22.5 → 17.1 → 15.6 → 20.8
Final loss: ~20
```

No non-training methods (no retrieval, no tool use, no prompting scaffolding).

---

## 4. Results


**Regret Ranking — All Models (lower = better)**

| Rank | Model | Regret |
|---|---|---|
| 1 | SFT Tic-Tac-Toe | **0.000** |
| 4 | Combined Tic-Tac-Toe | **0.000**|
| 8 | Combined Chopsticks |  **0.000**|
| 3 | RL Chopsticks | 0.100 |
| 5 | SFT Chopsticks | 0.291 |
| 6 | RL Tic-Tac-Toe | 0.327 |
| 9 | RL Connect 4 | 0.426 |
| 2 | SFT Connect 4 | 0.621 |
| 7 | Combined Connect 4 | 0.800 |

**Average game length by approach:**

![Average Game Length](https://github.com/bu-ds595/final-project-localminimaclub-1/blob/4464ec9b504626012f4098e335d5a830de0cdbed/example%20outputs/avg_game_length.png)

SFT Connect 4 produced the longest average games (9.0 moves), suggesting the model learned sustained strategic play. 

**Illegal move rate:** All trained models achieved **0% illegal moves** across all games. The base model regularly made illegal moves; fine-tuning eliminated this completely without any explicit illegal-move penalty in the training signal.

### Did fine-tuning help?

Yes — significantly on small exhaustive games, moderately on larger games.

- **Chopsticks Combined:** 100% Winrate — perfect optimality. Chain-of-thought format was sufficient. Whereas just RL had an 80% winrate.
- **Tic-Tac-Toe SFT:** 0.000 regret — perfect. Exhaustive oracle data with full state coverage.
- **Combined on Connect 4:** 0.800 regret — weakest result. RL was harder to stabilize for the larger branching factor.

### Failure modes and regressions

- **Connect 4 underperformed** (0.800 vs 0.426). RL training appeared unstable — weight similarity analysis showed near-zero self-similarity for `rl_connect4`, suggesting possible weight collapse.
- **Chopsticks showed near-zero transfer** to Connect 4 and Tic-Tac-Toe. Backward induction learned on one game did not seem to generalize to others.
- **Combined connect 4 training did not outperform single-game training** and performed worse, might need different weighting or other steps to improve results.

### Weight Similarity Analysis

We computed cosine similarity between LoRA weight matrices across all 6 trained models (SFT, RL, Combined × 3 games):

![Weight Similarity Heatmap](https://github.com/bu-ds595/final-project-localminimaclub-1/blob/7de442feec2db188cb857270232b90ca03e779bc/example%20outputs/weight_similarity.png)

Key observations:
- All models show relatively high similarity (0.67–0.80), suggesting some shared game-reasoning representation
- `sft_connect4` ↔ `rl_chopsticks`: 0.800 — highest cross-model similarity
- Some correlation between game weights confirms the hypothesis that game-theoretic training shares underlying structure

### Transfer Learning Results

![Transfer Win Rate](https://github.com/bu-ds595/final-project-localminimaclub-1/blob/816772ca2d35dba7e8fc8b9820d7a53d3a7b6fb2/example%20outputs/transfer_results.png)

- **Connect 4 → Chopsticks:** Transfer win rate ~0.6 — strong positive transfer
- **Tic-Tac-Toe → Chopsticks:** Transfer win rate ~0.7 — moderate transfer except for SFT
- **Chopsticks SFT → other games:** Near zero transfer

Winning is asymmetric as Chopsticks is the easiest game to start to win at, tictactoe usually ends up as a draw if played well which is not displayed here. Random choices in chopsticks give a resultant 50% win rate. 

---

## 5. Token Economics

| Run | Model | Method | Examples/Episodes | Est. Tokens Trained | Approx. Cost |
|---|---|---|---|---|---|
| Chopsticks | Llama 3.1-8B |  2 epochs | 979 | ~200k | ~$0.50 |
| Tic-Tac-Toe | Llama 3.1-8B | 2 epochs | 7,684 | ~1.5M | ~$3.00 |
| Connect 4 | Llama 3.1-8B |  2 epochs | 17,000 | ~3.5M | ~$8.00 |
| **Total** | | | | **~7.2M** | **~$11.50** |

Budget allocated: $100. Budget used: ~$26–30. Remaining budget preserved for re-runs.


---

## 6. Conclusions

### What worked

- **SFT with chain-of-thought oracle completions achieves perfect optimality for small exhaustive games.** Both Chopsticks and Tic-Tac-Toe reached 0.000 regret in some models. The key was combining both SFT & RL, this gave the model something to learn off of and then forced it to learn through scoring.
- **Zero illegal moves across all training approaches.** Fine-tuning reliably taught legal move generation without any explicit penalty. This was a consistent win across all methods and games.
- **SFT outperformed RL for Connect 4** (0.100 vs 0.800 regret). Pre-computed oracle labels were more reliable than sparse RL rewards for a game with high branching factor.

### What needed changed

- **It unstable for Connect 4.** The reward signal is too sparse when the game lasts 30+ moves. Training appeared to collapse based on weight similarity analysis.
- We needed to have transfer regret analysis to figure out if the model was comparable as wins are not a great measurement.
- We should have tested the model even more on different levels, maybe make it also play against a random bot to test if it was actually able to win. 

### What we'd do differently

**With 10x compute:**
- Deeper alpha-beta oracle (depth 4–6) for Connect 4 to get higher-quality training labels
- Curriculum RL: start vs random bots, gradually increase opponent strength
- Formal OR scoring of frontier and base models on the full eval set

**With 1000x compute:**
- Use a perfect solver for Connect 4 (it is a solved game — first player always wins with correct play) to give exact oracle labels like Chopsticks
- Full cross-game transfer experiment: train on 2 games, test on the other 1
- Test against opponents of varying strength, not just the strongest possible algorithm
- More RL episodes with larger episode batches for stable GRPO
- Add a checkers model to be able to test a significantly harder game

### Interesting findings 

1. **Zero illegal moves was a surprise.** We expected to need an explicit illegal-move penalty in the reward signal. Every training method eliminated illegal moves on its own — the models learned legal move generation as a side effect of learning to play well.

---

## 7. AI Usage

We used Claude (Anthropic) extensively throughout this project:

- **Dataset generation:** Python minimax and alpha-beta solvers written with Claude assistance. Debugging game tree logic, handling cyclic states in Chopsticks, and generating SFT-format completions.
- **Eval scripts:** Automated scoring scripts built with Claude, including the minimax oracle re-implementation for runtime scoring.

All experimental results — loss curves, regret scores, win rates, weight similarity values, transfer results — are real numbers from actual training runs on Tinker. No results were fabricated or estimated without explicit disclosure.

---

## Repository Structure

```
/
├── Dataset/
│   ├── chopsticks_train.jsonl    # 979 training examples
│   ├── chopsticks_eval.jsonl     # 173 eval examples (held out)
│   ├── ttt_train.jsonl           # 7,684 training examples
│   ├── ttt_eval.jsonl            # 1,356 eval examples
│   ├── c4_train.jsonl            # 17,000 training examples
│   └── c4_eval.jsonl             # 3,000 eval examples
├── env/                          # Game environments
├── example_outputs/              # Example outputs from prior LLMs
├── graphs/                       # Outputs as plots
├── marimo_notebook/              # Practice on working tinker
├── results/                       # Outputs as text
├── training/
│   └── connect4/                 # Connect 4 Marimo RL notebook
├── main_training_pipeline.py     # Main training file to run
├── game_utils.py                 # Main utils used for game state
├── run_pipeline.py               # test pipeline file to run
├── test_tinker_integration.py    # tests if tinker is connected
├── PROPOSAL.md
├── TINKER.md
└── README.md
```
# Making-LLM-s-Better-at-Games
