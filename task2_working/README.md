# Task 2 working: image parser

Goal: turn the rendered image into a GridState, even for asset looks not in `data/assets`.

| File | What it is |
|---|---|
| `gen.py` | makes synthetic training cells from the public assets (renderer-accurate, with colour jitter) |
| `parser_core.py` | the parser: cell features -> small neural net -> constraint-aware decision |
| `eval_parse.py` | scores a parser against the true cells of the public levels, using the real renderer |
| `holdout_test.py` | trains with some looks hidden, tests on the hidden looks (generalisation evidence) |
| `train_final.py` | trains on all looks, writes `paste_A_above_class.py` |
| `paste_B_agent_methods.py` | the Agent methods to replace |

Design decisions (from earlier experiments): one label per cell (softmax over 13 entities + floor);
edge-direction features so brick walls are not read as lava; hue/brightness jitter for unseen looks;
decisions use the task's guarantees (1 agent, 1 exit, #keys = #doors, doors locked at turn 0)
and cells that look identical share a label.

Earlier results (held-out looks, public levels drawn by the real renderer): 25% of looks held out ->
every cell right; 40% held out -> ~95% of levels read perfectly.
