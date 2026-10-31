# M0: Cloud training of the research reference model
<!-- label: ready-for-human -->
<!-- blocked-by: 09 -->

## What to build

Full-scale training of the research reference model on the lab GPU cluster (Slurm): Backbone, then the Versatile model, on the shaped AMASS Clips with all M0 options, then evaluation protocol v2 on the Test clips. The agent writes the submit scripts and hands over the exact commands; the maintainer runs every transfer and every job. The cluster specifics come from the maintainer's private how-to, which is kept out of the repo.

## Acceptance criteria

- [ ] Submit scripts follow the cluster's helper-based pattern from the private how-to (GPU flag, environment exported from the submit shell, submitted from the results directory, commit hash logged).
- [ ] Nothing cluster-specific is committed: usernames, hostnames and cluster paths come from a gitignored local config or environment overrides, so the public repo only holds generic scripts.
- [ ] Jobs are resumable: rerunning the same submit command after a timeout or cancel continues from the last checkpoint and skips finished work.
- [ ] Ticket ends with a hand-over for the maintainer: data rsync command(s), branch to pull, environment to activate (and any packages to add), exact submit commands with overrides, how to follow the jobs, how to check they finished, and the rsync that brings results back.
- [ ] A results .json and .md on the Test clips, in all Masking modes, centered and wild cameras, with noise on.
- [ ] Parameter count and `benchmark.py` latency are reported.
- [ ] A short README section is added.
