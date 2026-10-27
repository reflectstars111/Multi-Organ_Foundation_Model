# Q01 Organ Weighting Implementation Plan

> **For agentic workers:** Use parallel implementation of independent modules, then integration and independent review. The user has explicitly authorized implementation and launch; no additional plan approval is required.

**Goal:** Launch the 27 paired, resumable Q01 runs and their deferred test/comparison stages.

**Architecture:** Add an independent training entry point using the existing models, targets and sampler. Freeze train-only weights, source, data content, full draws and diagnostic/selection rules before two GPU workers consume the experiment queue.

**Tech Stack:** Existing project conda environment, PyTorch, NumPy, pytest, Linux flock.

**Spec:** `docs/EXPERIMENT_QUEUE.md` (Q01).

## Global Constraints

- 3 models × 3 arms × seeds 42/43/44; all runs start from scratch.
- 512×512, 80 epochs, base24, effective batch16/microbatch4, AdamW 1e-4/weight decay1e-4, constant LR, float32, clip5, workers0, no augmentation or auxiliary losses.
- 500 images/domain/epoch, 4500 total, 282 optimizer updates; paired initialization and draws.
- Train-only statistics, existing paper5/trainmore_v6/ct7_2 records hash `f621896f10234777d3067a019d96a7cb1dae3b6add305bca32815298744315e0`.
- Keep historical source and results intact; new files only except queue/documentation updates. Do not change RadarVLA. Do not commit unrelated workspace changes.
- Existing GPU locks and free-memory admission; no interference with other processes. No continuous assistant monitoring after launch.

## Review Focus

- Ignore pixels and empty targets must not create spurious gradients or undefined weights.
- A/B/C must preserve image denominators and match microbatch/full-batch gradients.
- Diagnostics must not consume training RNG or mutate weights, buffers or gradients.
- Restart must reject changed source/data/configuration and incomplete artifacts must not be mistaken for completion.
- Test evaluation must wait for all 27 training/validation selections; missing guardrail candidates must remain explicit.

### Task 1: Train statistics and weighted losses

**Files:** `unet_moe/organ_weighting.py`, `tests/test_organ_weighting.py`.

**Interfaces:** `collect_statistics(dataset) -> dict`; `loss_components(logits,target,arm,weights) -> dict`; `weighted_loss(...) -> Tensor`.

- [x] Write and run failing tests for A equivalence, B equal expected coefficients, C bounded normalized BCE, ignored pixels, empty targets and accumulation.
- [x] Implement streaming train statistics, source/group counts and frozen weight arrays.
- [x] Run `.conda/bin/python -m pytest tests/test_organ_weighting.py -q`; expected all pass.

### Task 2: Evaluation and diagnostics

**Files:** `unet_moe/organ_diagnostics.py`, `tests/test_organ_diagnostics.py`.

**Interfaces:** `evaluate(model,kind,loader,device,groups=None)`, `guardrail_passes(metrics,config=None)`, `diagnose_gradients(model,kind,dataset,indices,device,arm,weights,micro_batch_size=1)`.

- [x] Test per-source aggregation, failed organ guardrails and diagnostic RNG/state/gradient preservation.
- [x] Implement one-pass evaluation and per-organ encoder gradient probes.
- [x] Run `.conda/bin/python -m pytest tests/test_organ_diagnostics.py -q`; expected all pass.

### Task 3: Frozen protocol and training

**Files:** `unet_moe/organ_weighted_suite.py`, `tests/test_organ_weighted_suite.py`.

**Interfaces:** prepare, train/resume, verify-run, preflight, evaluate-test, summarize CLI modes; common `--output`, run identity `--kind --seed --arm`.

- [x] Test sampler/initialization identity, restoration of full training state, strict frozen protocol validation, and paired delta summary.
- [x] Freeze complete draws and per-epoch hashes, file-content manifest, initialization hashes and source snapshot; define C and guardrail constants before training.
- [x] Implement training, best/last/guardrail checkpoints, fixed-epoch diagnostics, final/best/guardrail test evaluation and paired summary.
- [x] Run targeted integration tests and medical regression suite; expected all pass except documented optional-environment skips.

### Task 4: Scheduling and launch

**Files:** `scripts/run_organ_weighted.py`, `tests/test_organ_weighted_controller.py`, `docs/Q01_ORGAN_WEIGHTED_512.md`, `docs/EXPERIMENT_QUEUE.md`.

**Interfaces:** controller invokes suite modes from frozen source and validates `protocol_sha256`; reads `preflight.json` with per-model memory requirements.

- [x] Test queue uniqueness, GPU locking/admission, restart, failed subprocess propagation, and training-before-test barrier.
- [x] Preflight two real worst-batch training updates per architecture with C weights, and diagnostic execution; save measured memory plus reserve.
- [x] Independently review integration and fix material findings with regression tests.
- [x] Launch detached controller from source snapshot, verify live worker processes and first training progress; publish exact status/log/resume commands.

## Frozen analysis choices

C modifies BCE on positive image/channel pairs only: q=(foreground_positive+1)/(valid_positive+2), r=clip(sqrt((1-q)/q),1,4), z=1+(r-1)q; foreground factor r/z, background factor 1/z. Empty-channel BCE and soft Dice are unchanged. B's organ factor applies to the total channel loss.

Guardrail candidates must have abdomen macro positive Dice≥0.40, all five organ positive Dice≥0.10, vessel Dice≥0.20, each organ positive zero-overlap rate≤0.50 and empty false-positive area fraction≤0.02. Missing required metrics fails the guardrail. These are prespecified research diagnostics, not clinical criteria.

Diagnostics use the first positive and first empty training image for each structure, deduplicated, at epochs 0/1/10/20/40/60/80. They report per-image gradient norms, not a falsely inferred joint gradient norm.

## Verification record

2026-10-07: final medical regression `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .conda/bin/python -m pytest tests -q`: 224 passed, 2 skipped (57.36 s). Training/resume integration reproduces final parameter SHA and loss history exactly; all original medical tests remain included. Independent review by the loss, diagnostics and controller implementers found no launch blocker after fixing resumed-history validation, CPU checkpoint loading and fixed-probe validation. No Git commits or RadarVLA changes were made.

Launch: all three GPU preflights passed; controller PID 2123736 launched the 27-run matrix from the frozen source snapshot. GPU0 started image_mean/shared/seed42; GPU1 started organ_exposure/shared/seed42. Remaining epochs, test evaluations and paired outcomes are asynchronous experimental work, not completed results.
