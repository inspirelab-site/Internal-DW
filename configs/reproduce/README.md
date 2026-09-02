# Paper reproduction configurations

These files record the canonical settings used by the released paper
checkpoints. Reader-facing launchers source them instead of duplicating the
settings inside multiple shell scripts.

`PAPER_GLOBAL_MICROBATCH` and `PAPER_GRAD_ACCUM_STEPS` define the optimization
schedule. On multiple GPUs, the launcher divides the global microbatch evenly
over DDP ranks while retaining the recorded accumulation count. For example,
the MG Full-BPTT setting maps from `1 x 32 x 1` to `4 x 8 x 1`; the MG
Internal-DW setting maps from `1 x 4 x 8` to `4 x 1 x 8`.

Readers may override `BATCH`, `GRAD_ACCUM`, and the other documented variables
at launch time. Such overrides are useful for hardware constraints but are no
longer the exact paper training schedule.
