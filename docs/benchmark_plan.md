# Benchmark design (not locked; no confirmation access)

As checked on 2026-10-01, the official leaderboard source reports
[TabArena-v0.1.9.3, dated 2026-09-28](https://huggingface.co/spaces/TabArena/leaderboard/raw/main/website_texts.py).
The [official repository](https://github.com/autogluon/tabarena) recommends first
developing on TabArena and then testing BeyondArena, which includes grouped and
temporal tasks. The leaderboard version is not a dataset/code version: record
repository revision, curation version, task IDs, split IDs, and metric conventions
separately when importing the benchmark. Recheck before locking the study.

`configs/datasets/benchmark_policy.yaml` records scope and unfulfilled gates.
Built-in sklearn tasks are implementation checks only, not a substitute for an
established benchmark suite. No benchmark task has yet been excluded or tested.
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

