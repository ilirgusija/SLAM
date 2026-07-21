# Handoff: finite-memory info MDP (`info-mdp`)

**Date:** 2026-07-21  
**Branch:** `info-mdp`  
**Next step:** exercise `notebooks/finite_memory/finite_memory_q_learning_mapping.ipynb` on a GPU/HFMS node (same filesystem).

## Chat exports (this directory)

| File | What |
|------|------|
| `INFO_MDP_HANDOFF.md` | This summary |
| `info-mdp-finite-memory.chat.md` | Readable user/assistant transcript |
| `info-mdp-finite-memory.chat.jsonl` | Raw Cursor transcript (~370 KB) |
| Cursor id | `f496c147-9580-4c68-907f-4f48dc201908` |

## Package layout

```
src/classes/           # shared: models, maps, POMDP, costs
src/belief_quantized/  # BeliefMDP_n*, BeliefMDP_n_M*, value iteration
src/finite_memory/     # window, InformationMDP, ApproximateInformationMDP,
                       # MappingRollout + FiniteMemoryQLearning
```

**Removed as bloat:** `factory.py` / `MappingBackend`, `env.py` / `MappingInfoEnv`.  
Rollout lives in `q_learning.py` as `MappingRollout`; MDPs take `BeliefMDP_n_Mapping` directly.

## Design invariants

- Stage cost: shared `classes.costs.stage_cost` → \(\hat\rho(h,u)=\rho(\Psi(b^*,h),u)\).
- Filter: `BeliefMDP_n_Mapping.F` only (no duplicated Bayesian update).
- Vertical slice: known-pose active mapping, finite `LandmarkMap`.
- Tests: `USE_CUPY=false /path/to/venv/bin/pytest test/finite_memory/ -q` → 15 passed.

## Notebook to run next

`notebooks/finite_memory/finite_memory_q_learning_mapping.ipynb`

Uses tiny alphabets (`pose_n=map_n=obs_n=action_n=2`, `memory_N=0`), `MappingRollout` + tabular Q-learning. On GPU, set CuPy if desired (`USE_CUPY=true`) once the tiny slice is green; keep alphabets small — \(|D^N|=|Y|^{N+1}|U|^N\`.

## Venv on this cluster

`/global/home/hpc5656/venv_pomdp`

## Intentionally not in the last commit

Unrelated dirty notebooks under `notebooks/{localization,mapping,slam}/`, `notebooks/*/outputs/`, and `docs/root.tex`.
