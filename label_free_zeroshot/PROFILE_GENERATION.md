# Generating the LLM-authored activity profiles

This documents how the per-class motion profiles in `pseudo_labels.py` were
produced, so the label-free / prototype experiments (paper Tables VII–VIII) are
reproducible. The profiles assign, for every activity class and every body-worn
limb, one of `{low, medium, high}` to each of the eight motion features.

## Model and settings

- **Model:** Claude (Anthropic), used interactively through **Claude Code**.
- **Interaction:** the model authored `pseudo_labels.py` directly as Python
  `ActivityProfile` objects, in a single pass, from the prompt below plus the
  feature definitions and the class names only. No sensor data was shown to the
  model, and the assignments were used verbatim (no manual editing of levels).
- **Decoding:** Claude Code defaults (no custom temperature / sampling flags).

> The profiles are physical-intuition priors, not data-derived statistics: they
> encode which features a domain expert (here, the model) expects to be low /
> medium / high for each activity, relative to the range spanned by all
> activities in that dataset. Because they come from reasoning rather than
> measurement, an independent regeneration is expected to agree on the large
> majority of cells but not to be byte-identical (see "Reproduction" below).

## The eight features (given to the model)

| key | definition |
|-----|------------|
| `rms`              | mean RMS acceleration magnitude — overall motion intensity |
| `speed_mean`       | mean detrended integrated speed (m/s) |
| `distance_total`   | total path length from double-integrated linear acceleration (m) |
| `impact_rate`      | fraction of frames with \|\|acc\|\| > 2g (≈19.6 m/s²) — strike/landing |
| `tilt_mean`        | mean tilt angle from the gravity direction (rad); low = upright |
| `tilt_variability` | std of tilt within the window (rad) — oscillation / repetitiveness |
| `angular_range`    | max cumulative rotation from window start (rad) — range of motion |
| `angular_rate`     | mean absolute angular rate across axes (rad/s) — rotation speed |

Levels are percentile bands of each feature's distribution across all activities
in the dataset: `low` = 0–33rd, `medium` = 33rd–67th, `high` = above 67th.

## Limb layout (given to the model)

- **MM-Fit:** `wrist` (both smartwatches), `lower_body` (phone at hip). *(head/earbud optional, unused here.)*
- **WEAR:** `arm` (left+right arm), `leg` (left+right leg).
- **PAMAP2:** `arm` (hand IMU), `torso` (chest IMU), `leg` (ankle IMU).

## The prompt

> You are a human-activity-recognition expert. For each activity class below,
> describe its expected accelerometer signature by assigning **low**, **medium**,
> or **high** to each of eight motion features, **separately for each body limb**
> that carries a sensor. The levels are relative to the range spanned by *all*
> activities in this dataset: `low` = bottom third, `medium` = middle third,
> `high` = top third of that feature's distribution.
>
> The eight features are: `rms` (overall intensity), `speed_mean`,
> `distance_total`, `impact_rate` (fraction of high-acceleration strike frames),
> `tilt_mean` (posture angle from vertical; low = upright), `tilt_variability`
> (oscillation), `angular_range` (range of motion), `angular_rate` (rotation
> speed).
>
> The limbs for this dataset are: <limb list>. The activity classes are:
> <class list>. Reason from the biomechanics of each exercise: which limb drives
> the motion, whether it involves foot strikes / impacts, the held posture, and
> the amount of rotation. Reflect the contrasts between classes (e.g. arm-driven
> vs leg-driven exercises, static holds vs dynamic movements).
>
> Emit the result as `ActivityProfile` Python objects: one per class, with a
> `limbs` dict mapping each limb to `{feature: level}` for all eight features,
> plus a short `notes` string explaining the discriminating signature.

The exact class lists per dataset are the keys of `MMFIT_ACTIVITIES`,
`WEAR_ACTIVITIES`, and `PAMAP2_ACTIVITIES` in `pseudo_labels.py`.

## Reproduction

`reproduce_profiles.py` quantifies how well a regenerated set of profiles matches
the committed ones. Run a fresh Claude (ideally in a clean session, so it has not
seen `pseudo_labels.py`) on the prompt above, save its assignments as
`regenerated_profiles.json` (schema: `{"<dataset>/<label>": {"<limb>":
{"<feature>": "low|medium|high"}}}`), then:

```bash
python reproduce_profiles.py regenerated_profiles.json
```

It reports, per dataset: exact-match rate over all (class × limb × feature)
cells, and the ordinal (off-by-one) distance, treating low<medium<high.

A committed sample regeneration for MM-Fit is included as
`regenerated_profiles.json`. Scored against the committed profiles it reaches
**97.2% exact match** over the 176 cells (11 classes × 2 limbs × 8 features),
mean ordinal distance **0.028** — i.e. the handful of disagreements are all a
single band apart (medium↔high), never low↔high. This confirms the prompt plus
feature definitions recover the profiles up to the expected medium/high
subjectivity. (That sample was generated with the originals present in the
working tree; rerun in a clean session for a stricter independent check.)
