#!/usr/bin/env bash
# Canonical configuration recorded by the MG Internal-DW paper checkpoint.

PAPER_DATASET=mackey_glass
PAPER_CONDITION=tau30
PAPER_TRAIN_HORIZON=32
PAPER_EVAL_HORIZON=48

# The reference run used one GPU, eight 4-example microbatches per update.
# Keeping the microbatch and accumulation schedule matters because the DW
# controller calibrates once per microbatch.
PAPER_WORLD_SIZE=1
PAPER_GLOBAL_MICROBATCH=4
PAPER_GRAD_ACCUM_STEPS=8

PAPER_EPOCHS=100
PAPER_EARLY_STOP_PATIENCE=20
PAPER_LEARNING_RATE=1e-4
PAPER_OPTIMIZER=adam
PAPER_SGD_MOMENTUM=0.0
PAPER_WEIGHT_DECAY=1e-4
PAPER_GRAD_CLIP=1.0
PAPER_SCHEDULER=step
PAPER_NUM_WORKERS=4

PAPER_MAX_ORIGINS_PER_ITEM=64
PAPER_ORIGIN_BATCH=16
PAPER_BOOTSTRAP_DRAWS=10000
