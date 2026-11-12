# CS2109S Maze Agent — Grid Adventure V1

An agent that plays **Grid Adventure V1**, the AY2026/27 Sem 1 capstone project for NUS CS2109S (Introduction to AI and Machine Learning). The agent navigates grid mazes full of walls, lava, keys, locked doors, boxes, coins, gems and power-ups, and tries to reach the exit with the best possible score inside a 150-turn limit.

It handles three tasks with **one `Agent` class**:

| Task | What the agent sees | What it needs |
|---|---|---|
| 1 | The full game state (`GridState`) | Plan the best route |
| 2 | Only a rendered **image** of the grid | Read the image into a state, then plan |
| 3 | Boss level (many mechanics at once, state or image) | Plan under a bigger search space |

## Results on the public levels

The course's public smoke checks (22 gameplay levels in both observation modes, plus the two boss levels):

| Check | Result |
|---|---|
| Task 1 (state), 22 levels | 22/22 wins, performance 1.0 on every level |
| Task 2 (image), 22 levels | 22/22 wins, performance 1.0 on every level |
| Task 3 (boss), both levels, state and image | 1.0 on the first boss level, about 0.99 on the second |
| Image parser on **unseen** sprite looks (25% of looks hidden from training) | Every cell read correctly in 192/192 re-rendered public levels |

"Performance" is the course's score: 1.0 means the agent matched the reference optimum, and 0 means a failure or timeout. These are local smoke checks, not a grade; the real tests use hidden levels.

## How it works

### Planner (Tasks 1 and 3)
- **State-based A\*** over the game's own `step` function, so every rule (keys, doors, lava, shield, ghost and speed power-ups, boxes, coin and gem rewards) is simulated exactly rather than re-implemented.
- A **heuristic** that lower-bounds the remaining cost, and **pruning** of moves that cannot help.
- A **three-pass weighted A\* safety net** (`PASSES = ((1, 2.5), (3, 3.3), (10, 4.0))`): it first searches for the optimal plan, and if time runs short it falls back to faster, greedier searches so the agent always returns a winning plan rather than timing out. Planning stops after about 4 seconds.
- The plan is computed **once** on the first turn and then replayed action by action.

### Image parser (Task 2)
The image is cut into cells, each cell is classified, and the result is turned into a `GridState` that the same planner can use.
- **Model:** a linear softmax classifier (scikit-learn `LogisticRegression`) over 14 classes: the 13 entity types plus floor. Features are a 16×16 colour thumbnail, edge-direction energy (this is what tells brick walls from lava), colour statistics, and a shape mask measured against the level's floor tile.
- **Two stages:** the first stage finds the floor tile of the level, and the second classifies every cell relative to it.
- **Unseen looks:** every level draws entities with a different "look". The training set is synthetic: public sprites composed the way the renderer composes them (background, then entities, then corner icons), with colour jitter and mirroring. Evaluation hides some looks completely and checks that they are still read correctly.
- **Game rules as constraints:** exactly one agent and one exit, as many keys as locked doors, doors locked at turn 0, and cells that look identical must share a label.
- **Hardcoded model:** the course submission has no auxiliary files, so the trained models are embedded in the agent cell using the course helper `generate_sklearn_loader_snippet` (zlib-compressed pickle).

## Repository layout

| Path | What it is |
|---|---|
| `capstone-project.ipynb` | **The main notebook.** The course's instructions and public test harness, with my agent in the Implementation section |
| `scratch.ipynb` | The Task 2 development notebook: dataset generation, features, model comparisons, held-out evaluation, and export of the trained models |
| `task2_models_snippet.py` | The trained floor and cell classifiers, as the generated `get_floor_model()` and `get_cell_model()` loaders pasted into the agent cell |
| `task2_dataset_sample.npz` | A sample of the generated training data |
| `heldout_assets_scratch/` | The sprite looks held out from training, used for the generalisation tests |
| `utils.py`, `public_levels/`, `data/assets/` | Course-provided helpers, public levels and sprite assets |
| `environment.yml` | Environment definition |
| `paste_task2_parser.py`, `task2_model_b64.txt`, `task2_working/` | Earlier, superseded Task 2 attempts, kept only as a record of the work |
| `tutorial.ipynb`, `trst_env.py` | The course tutorial notebook and a small interpreter check |

## Run it yourself

You need **conda or mamba** and **git**. Python 3.13 is installed by the environment file.

1. **Clone and create the environment**
   ```bash
   git clone https://github.com/vismaymenon/CS2109S_Maze_Agent.git
   cd CS2109S_Maze_Agent
   conda env create -f environment.yml
   conda activate cs2109s-ay2627s1-capstone-project
   ```
   This installs the `grid-universe` and `grid-adventure` game packages straight from GitHub, so it needs internet access. It also installs PyTorch, which this agent does not use, so the install is larger than it needs to be.

2. **Start Jupyter and open the notebook**
   ```bash
   jupyter notebook capstone-project.ipynb
   ```
   Select the `cs2109s-ay2627s1-capstone-project` kernel if prompted. In VS Code, open the notebook and pick that environment as the kernel.

3. **Define the agent.** The Implementation section has two cells that define `Agent`. The first is an earlier version that only handles Task 1, and the second is the final agent: the long cell that contains the embedded models and the `ImageParser`. Run the second one last, since it replaces the first.

4. **Run the public checks.** Run the evaluation-helper cell, then the check cells for each task:
   - **Task 1:** the cell with `RUN_TASK_1_PUBLIC_CHECKS = True`
   - **Task 2:** the cell with `RUN_TASK_2_PUBLIC_CHECKS = True`
   - **Task 3:** the cell with `RUN_TASK_3_PUBLIC_CHECKS = True`

   Each shows a table of results per level and a summary with wins, timeouts and average performance. To run all 48 checks at once, set the three task flags to `False`, set `RUN_ALL_PUBLIC_CHECKS = True` and run that cell.

   Expect Tasks 1 and 2 to take a few minutes and the boss levels longer, since the planner may use its full time budget there.

5. **Optional: watch the agent play.** To step through one boss-level run frame by frame, set `SHOW_BOSS_TRACE = True` in the "Debugging" section and run that cell. This uses `ipympl`, so it works in classic Jupyter and is flaky in some other front ends.

   To run the agent on any single level from a cell:
   ```python
   from public_levels.intro import build_level_boss
   from utils import create_env, evaluate

   env = create_env(build_level_boss, observation_type="image", seed=42)  # or "gridstate"
   total_reward, win, lose, history = evaluate(Agent, env)
   print(total_reward, win, lose)
   ```
   Changing `seed` changes the look of the sprites in `"image"` mode, which is a good way to test the parser on new looks.

### Troubleshooting
- **`ModuleNotFoundError: grid_adventure`:** the environment is not active, or the pip install from GitHub failed. Re-run step 1 with a working connection.
- **A timeout reported locally:** the local time limit is only diagnostic and does not interrupt the run, so a slow machine can show timeouts that the grader would not. Treat repeated timeouts on a fast machine as real.
- **Version errors when loading the models:** they were pickled with scikit-learn 1.7.0, so use the pinned `environment.yml` rather than a newer scikit-learn.

## Acknowledgements
Game environment, public levels, assets and evaluation helpers are provided by the CS2109S teaching team (`grid-universe`, `grid-adventure`, `grid-play`). The planner, image parser and training pipeline are my own work.
