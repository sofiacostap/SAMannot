# Lek3 intake — separate from the previous GT review

The new session format and annotation semantics are under investigation. No
propagation or evaluation configuration has been selected. Historical Lek4 outputs
and GT-review flags are not inputs to this experiment.

Run from the repository root:

```bash
python experiments/lek3/inspect_session_stream.py --session "/path/to/session.pkl"
```

The inspector uses only the Python standard library. It reads pickle opcodes and
seeks over binary image/array payloads without unpickling or importing referenced
classes. It never loads the large data buffers into memory. A new, small JSON
report is written under experiments/lek3/results. The source is read-only.

The report records observed strings, nearby scalar fields, payload lengths and
STOP positions. These are storage observations, not validated session contents:
memo references are not resolved, dictionary nesting is not reconstructed, and
prompt provenance and mask coverage cannot be concluded from this scan. It can
scan multiple consecutive pickle objects such as backend and frontend session
records. Unexpected or truncated arguments stop the scan and produce a report.
The scan does not prove successful transfer: use producer checksums/confirmation
when available. File size and modification time are checked before and after.

Only code and small reports belong in Git. Keep videos and annotation files on
the data drive. No GPU or additional packages are required for this inspection.

Tests: `python -m unittest discover -s experiments/lek3 -p 'test_*.py'`.

## Small alignment pilot

`alignment_pilot.py --session /path/to/session.pkl --video /path/to/video.mp4`
exports saved point/box records and seven sampled raw GT mask sets. It compares
each set against several candidate video indices using sequential decoding.
Results are written to a new `experiments/lek3/results/pilot-*` directory.
Requires NumPy, Pillow and OpenCV; no GPU or SAM2 inference is used.

The packed masks are unpacked without interpolation. Missing label entries are
reported as unknown rather than interpreted as absent birds. Preview colours
are listed in the report. The current pilot is specific to the inspected Lek3
coverage and stops on unsupported storage schemas or mismatched dimensions.

Offsets 0, 1, block+2 and 28 are exploratory comparisons, not an accepted mapping.
The block-dependent candidate tests possible extraction drift; the constant 28
candidate comes from the cached-overlay search. No input files are changed.
Saved prompts may include later corrections; their original annotation provenance
and correspondence to video indices remain unverified. Do not select experimental
anchors or calculate evaluation scores from these candidates before review.
