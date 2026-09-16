# Task 5: human-reviewed identity-swap prototype

Status (16 September 2026): prototype tested and Lumen suggestions analysed.
Six proposals remain pending visual approval.
No correction has been approved or applied to project data. See [findings](FINDINGS.md).

## Purpose and scope

Find intervals where two predicted masks may cover the right birds but carry
each other's identities. A label swap changes mask ownership only: it cannot
repair a wall mask, missing bird, inaccurate boundary or arbitrary three-way
exchange. Original predictions and GT remain untouched.

Automatic suggestions in this prototype use the released GT and are **research
diagnostics, not a deployed GT-free identity detector**. The Task 4 A/B/C scorer
is not used as an authoritative swap detector. In a future no-GT interface,
a reviewer would supply/review identities using footage and annotation context;
automatic GT-free detection is still separate work.

## Propose and visually inspect

`identity_swap_review.py propose` checks the closed Task 2/4 inputs and the
exact Task 4 baseline prediction hashes. For each frame it compares independent
prediction masks with the trusted, present GT identities. It proposes a pair
only when both cross-identity IoUs reach 0.5 and both exceed their respective
current-identity IoUs by at least 0.2. Competing pairs sharing a bird are rejected
as ambiguous. These conservative discovery thresholds are not calibrated
deployment thresholds.

Consecutive evidence frames form an interval within one block. Annotation frames
are excluded; intervals never bridge unavailable/held GT or block boundaries.
The default minimum length is three frames. This can miss short, weak, complex
or GT-uncertain swaps. A result with no candidates does not establish no swaps.
The interval start/end describe the observed evidence, not independently proven
biological boundaries.

Each review image shows the same verified photograph in four panels:

1. Photograph without masks.
2. Current prediction for the selected pair.
3. Trusted GT for that pair (held/unavailable GT is explicitly marked).
4. Proposed identity assignment, with exactly the same mask shapes.

Default samples are before the interval, start, midpoint, end and after it.
Context samples outside the proposed interval remain unchanged in panel 4.
Before approval check bird identity, GT plausibility and both interval boundaries.
Samples do not prove every intervening frame; leave uncertain proposals pending.
This is a research visual-review prototype, not yet an integrated biologist UI.

`index.html` lists all proposed events. Initially the 12 longest intervals receive
previews to keep the first review bounded. Other event previews can be rendered
explicitly with the `preview` subcommand; no new propagation is needed.

## Explicit approval and non-destructive correction

The `apply` subcommand requires named event IDs and a reviewer. It rejects
unknown IDs, unrendered events, duplicate approvals, overlapping/conflicting
identity changes, intervals crossing a block boundary and annotation-frame
changes. Do not run it until the human has approved those exact events.

It creates a **new identity overlay**, not a replacement propagation run:

- Only affected bird/frame PNGs are copied into the new output, byte-identically
  from the opposite identity. There is no mask editing or SAM2 rerun.
- `mask_sources.csv` resolves every bird/frame to its corrected or unchanged
  baseline file, with original source identity and hash. Consumers must use this
  table: the overlay's mask folder alone is intentionally incomplete.
- `foreground_confidence.csv` moves B with its source mask while retaining the
  target identity and correct frame/block metadata.
- `approved_events.json`, `corrections.csv` and `verification.json` record the
  reviewer, exact approved intervals, changed observations and provenance.
- A/C and combined scores must be recomputed if this corrected sequence is
  evaluated. Old temporal scores cannot be silently reused.

GT-assisted post-hoc corrections are not independent evidence that the scorer
has improved. Keep the Task 4 baseline unchanged and label any corrected-run
comparison separately. The Task 4 evaluator does not directly consume this
overlay; an explicit overlay-aware evaluation step is required later.

## First Lumen run

```bash
cd /media/hbvcai/HBVCSSD/samannot/SAMannot/SAMannot-audit
conda activate samannot
git pull --ff-only origin audit/lek4-validation &&
python -m unittest discover -s audit -p test_identity_swap_review.py &&
python audit/identity_swap_review.py propose
```

Upload the new proposal reports and JPG previews:

```bash
git add audit/results/swap-review-*/
git commit -m "Add Task 5 identity-swap proposals and visual previews"
git push origin audit/lek4-validation
```

Review `index.html` in the reported directory locally on Lumen; the same JPG
previews are uploaded to GitHub. GitHub displays the HTML source rather than
running the page. No project correction happens in this first run.

For additional previews, substitute actual returned IDs:

```bash
python audit/identity_swap_review.py preview --review REVIEW_DIRECTORY --events EVENT_ID
```

Only after explicit human approval of the reviewed IDs:

```bash
python audit/identity_swap_review.py apply --review REVIEW_DIRECTORY --approve APPROVED_EVENT_ID --reviewer "REVIEWER_NAME"
```

These placeholders are not runnable examples. No event is pre-approved.
