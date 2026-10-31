# Research-licensed data now, retrain on commercial-clean data later

The end goal is a commercial phone app, but AMASS, SMPL-X, BEDLAM, 3DPW, EMDB and BONES-SEED are licensed for research only (BONES-SEED also for qualifying startups). We use them freely during the research phase to validate architectures, and retrain the validated architecture on commercial-clean data before any product release: SOMA, MHR or Anny Body shapes, and motion from NVIDIA Open Model License generators (Kimodo-SOMA, MotionBricks) or licensed mocap. Weights trained on research-only data are never shipped.

## Consequences

- The training pipeline must not hard-depend on a research-only asset: anything taken from SMPL-X or AMASS (vertex mappings, key poses, motion seeds) must have a SOMA-native replacement, or the product retrain is blocked.
- Kimodo's SMPL-X checkpoint uses a research-only license; only the SOMA checkpoints count as commercial-clean.
- Research-only datasets stay usable for evaluation after the switch.
