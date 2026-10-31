# M0: Cloud training of the research reference model
<!-- label: ready-for-human -->
<!-- blocked-by: 09 -->

## What to build

Train the research reference model at full scale on the cloud platform, using sbatch files written to the maintainer's instructions: Backbone, then the Versatile model, on the shaped AMASS Clips with all M0 options. Then report it with evaluation protocol v2 on the Test clips.

## Acceptance criteria

- [ ] sbatch files follow the maintainer's cluster instructions, and the maintainer launches the jobs (or explicitly approves launching them).
- [ ] A results .json and .md on the Test clips, in all Masking modes, centered and wild cameras, with noise on.
- [ ] Parameter count and `benchmark.py` latency are reported.
- [ ] A short README section is added.
