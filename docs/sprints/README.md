# Sprints

Read `PRD.md` first, then `CLAUDE.md`, then `docs/ARCHITECTURE.md`. Then work these in order.

| # | Sprint | Gate | Effort |
|---|---|---|---|
| [00](00-foundation.md) | Foundation — config, seeding, scaffold | `pytest` green | 1 session |
| [01](01-data.md) | Data pipeline | Split + normalization tests pass | 1 session |
| [02](02-models.md) | Models & registry | Param counts exact; overfit test passes | 1–2 sessions |
| [03](03-training-engine.md) | Training engine | **SimpleCNN ≥ 85%** end-to-end | 2 sessions |
| [04](04-tracking.md) | Tracking & reporting | Two runs comparable; resume faithful | 1 session |
| [05](05-the-94-recipe.md) | The 94% recipe | **ResNet-18 ≥ 94.0%** | 1 session + ~5 h GPU |
| [06](06-augmentation.md) | Advanced augmentation | **≥ 95.0%** | 1 session + ~8 h GPU |
| [07](07-squeeze-and-analyze.md) | Squeeze & analyze | **≥ 95.5%** + model card | 1 session + ~4 h GPU |
| [08](08-hardening-release.md) | Hardening & release | All of M1–M7 green | 1 session |

Sprints 01 and 02 can run in parallel. Everything from 03 onward is sequential.

## How to run one

Give the assistant exactly three files — `CLAUDE.md`, `docs/ARCHITECTURE.md`, and the one sprint file — then paste that sprint's implementation prompt from the bottom of its page. More context than that makes the assistant wander.

Build components in the listed order, write each component's tests before moving to the next, and don't open the next sprint until the current gate passes. Commit per component as `sprint-N: <component>`.

## When a gate fails

Sprints 3 and 5 have debug checklists at the bottom of their files, ordered by how often each cause is actually the culprit. Work them top to bottom. Resist the urge to start changing hyperparameters — at this stage a missed accuracy target is nearly always a bug, not a tuning problem.
