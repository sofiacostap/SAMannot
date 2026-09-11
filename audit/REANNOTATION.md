# GT-assisted reannotation

This is a new experimental annotation set, separate from the original manual
baseline. No old pickle, GT, or prediction is edited. New points are saved in
`audit/results/reannotation-*/annotations.json`. GPU propagation integration is
a separate next step; this tool does not run inference.

Run on Lumen, from the audit checkout with the samannot environment active:

```bash
python audit/reannotate.py \
  --source audit/results/verified-source-20260911T164528649441Z \
  --prior audit/results/aligned-20260911T151902Z
```

Open http://127.0.0.1:8765 in a browser **on Lumen**. Keep the terminal running.
Use the block selector. Default candidate video indices are 374, 1124, 1874,
2624, 3374, 3781. These are midpoint suggestions, not certified good frames.
Enter a different index in the same block or use the +/-25 buttons.

Choose a sharp image showing all birds clearly. Compare the photograph with
the GT overlay. Colours correspond to GT IDs 14, 38, 75, 113, not historical
descriptive names. Verify identities visually; component counts, size and
border contact are warnings only and cannot prove GT correctness.

Select one GT ID at a time and click within that bird's body on either panel.
Both panels have the same verified pixels and coordinate system. Prefer a
clear interior location, away from legs, edges or overlapping birds. Repeat
for four birds. Tick the visual review box and save the block. Save requires
each point to fall within its selected GT mask, but that check alone does not
make an artefactual GT region valid. If GT is wrong, choose another frame and
record the suspect frame separately; do not click the artefact to pass validation.

The UI shows all saved block IDs after a successful save. Each save persists
immediately. Stop with Ctrl+C after saving. To resume in a later session, add
`--resume audit/results/reannotation-TIMESTAMP` using the exact directory printed
at startup. Changing to a saved block reloads its selected frame and points.

Upload the small annotation JSON through GitHub:

```bash
git add audit/results/reannotation-*/annotations.json
git commit -m "Add reviewed GT-assisted annotation points"
git push origin audit/lek4-validation
```

One annotation near the middle of each block is intended for propagation in
both directions from that same frame. Moving prompts from the start to the
middle changes the experiment. Under this design, block ends are farthest
from the prompt; any temporal-error analysis must use actual distance from
the chosen prompt. GT-guided clicks also make this a GT-assisted experiment,
not an independent estimate of ordinary user annotation performance.
