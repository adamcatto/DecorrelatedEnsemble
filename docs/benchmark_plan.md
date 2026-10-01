# Benchmark design (not locked; no confirmation access)

As checked on 2026-10-01, the official leaderboard source reports
[TabArena-v0.1.9.3, dated 2026-09-28](https://huggingface.co/spaces/TabArena/leaderboard/raw/main/website_texts.py).
The [official repository](https://github.com/autogluon/tabarena) recommends first
developing on TabArena and then testing BeyondArena, which includes grouped and
temporal tasks. The leaderboard version is not a dataset/code version: record
repository revision, curation version, task IDs, split IDs, and metric conventions
separately when importing the benchmark. Recheck before locking the study.

`configs/datasets/benchmark_policy.yaml` records scope and unfulfilled gates.
Built-in sklearn tasks are a small real development panel, not a substitute for
an established benchmark suite. Experiment 004 names breast cancer, wine and
diabetes before scored runs. No established-suite task has yet been excluded or tested.
Freeze a development subset by task metadata and compute limits before any
performance inspection. Reserve the remaining eligible tasks for confirmation;
publish every exclusion. Scope competitive current neural/foundation models from
official results, including model version and feasible dataset scale, rather than
assuming old TabPFN releases are the strongest baseline.

This algorithm performs its own model search/selection and should be scoped as a
TabArena **system** where appropriate; calling it a single untuned tree estimator
would misrepresent its compute. Under a shared tuning protocol, all extra nested
fits must count against the resource budget. Current synthetic static configurations
do not satisfy tuned, resource-matched benchmark comparisons.

## Baseline scope recheck during exp_006 (no benchmark scoring)

The official repository still distinguishes system-owned search from a shared
model protocol and recommends TabArena before BeyondArena. Its current core extra
lists TabPFN, TabICL, EBM, RealMLP, TabDPT and TabM. Rechecked the current
[leaderboard source](https://huggingface.co/spaces/TabArena/leaderboard/raw/main/website_texts.py)
and [TabPFN-3 technical report v2](https://arxiv.org/abs/2605.13986v2).
The report discusses substantially larger scales than older TabPFN versions;
its author-reported ranking is not our independent result. Do not use an old
small-N eligibility rule to omit current versions. The live leaderboard can
postdate that report; freeze exact checkpoints, code, access mode and resource
limits before confirmation. AutoGluon, competitive foundation/neural methods and
tuned boosting remain gates, not satisfied by exp_006's static classical controls.
No confirmation tasks or cached prediction arrays have been accessed.
