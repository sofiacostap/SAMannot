# Independent starting-frame experiments

This protocol tests ten alternative starting photographs across the five reviewed minutes.
Each run uses its own starting points and predictor states. The default propagation window
is 750 frames (30 seconds), centred on the start where possible and shifted at recording edges.
This is a configurable experimental choice. It does not propagate each run over all five minutes.
The earlier-time pass supplies frames in reverse order to the forward predictor in a separate state.

## Prepare on Lumen

From the repository root with the samannot environment active:

    python experiments/lek3/batch_start_experiments.py prepare --output experiments/lek3/start_experiments_v1

This reads the five reviewed session backups one at a time and fingerprints each.
It checks that each minute contains numbered images 000000–001499.
It creates plan.json and index.html with ten candidate prompt photographs.
Points come from the most interior pixel of each available reviewed mask.
This is a GT-assisted initialization experiment. It does not measure unaided manual prompting.

Open index.html. Check points, bird identities and the original review_notes.txt files.
Edit points or start indices in plan.json if necessary. If changing a start, inspect the new
photograph and replace every point accordingly; the preview image is the original proposal.
Birds missing or empty in the starting reference are unavailable for that run; missing entries
are not declared biological absence. Do not approve ambiguous identities or points.
For each accepted experiment change review_status to accepted, then set approved to true.
Unavailable birds are omitted, so runs can differ in their available identity set.

## First GPU run

    python experiments/lek3/batch_start_experiments.py run --output experiments/lek3/start_experiments_v1 --experiments 0

Run the remaining accepted experiments by omitting --experiments.
Completed runs are skipped. Interrupted runs retain a failed manifest; use a new experiment
output folder or deliberately rename the failed run directory before retrying.
Do not edit an approved plan after completing runs: create a new plan directory instead.

Each run stores independent binary PNGs, per-frame/bird confidence and area in scores.csv,
and run.json with settings, hashes, status and coverage.
The score is mean sigmoid over positive logits; empty predictions have no score.
Only starting-frame reference masks supply points. Later reference masks are not passed to SAM2.
The starting photograph must be excluded from later numerical accuracy summaries.

## Visual comparison

    python experiments/lek3/batch_start_experiments.py review --output experiments/lek3/start_experiments_v1

Open comparison.html. It samples every 25 frames (one nominal second), displaying the reviewed
reference beside each available run. Use --stride 1 for every frame, at substantially greater
storage cost. Global index 0 is source frame 30000, approximately player time 20:00.
Global index = (part number - 1) * 1500 + local photograph index.
Colours are consistent for bird1 red, bird2 green, bird3 yellow and bird4 blue.

The page compares overlapping windows where available; runs do not share a single complete
evaluation population. Combined overlays can hide overlap between individual bird masks.
Inspect the saved per-bird PNGs when assignment or boundary overlap is ambiguous.
The reference contains recorded uncertainty and missing entries. Establish tolerance/exclusion
decisions from reference evidence before computing comparative accuracy. This script does not
silently convert missing reference entries to absence, exclude difficult frames or calculate IoU.

## Verification and constraints

Local checks cover boundary windows and a synthetic visual-comparison exercise.
A real CUDA run is still required on Lumen. Sessions are trusted local pickle files.
The first real run should be checked visually before launching all ten.
Full-resolution images resized by the SAM2 loader can still consume substantial CPU RAM;
750-frame windows reduce this demand. Check free space and memory before a batch.

