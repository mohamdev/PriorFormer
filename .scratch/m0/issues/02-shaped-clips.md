# M0: AMASS Clips with real Body shapes
<!-- label: ready-for-agent -->
<!-- blocked-by: -->

## What to build

Preprocessing can write a second processed set of Clips with the AMASS Body shapes (betas) applied, separate from the template-shape set, and stores each Clip's Body shape and gender. This removes README Correction 7 (two template Body shapes) for new models. The paper's default stays unchanged.

## Acceptance criteria

- [ ] The shaped set keeps the same Clip names and train/test split as the template set.
- [ ] Each shaped Clip stores its Body shape and gender.
- [ ] A cheap check (std of thigh length across Clips) is clearly above zero for the shaped set and near zero within each gender for the template set; it runs as a script on processed data, and its logic is unit-tested on synthetic poses.
- [ ] The template-shape default and its outputs are unchanged.
