# Task 1 — image correspondence

All 3,802 FRAMES images matched uniquely and exactly to a sequential video decode.
GT image indices 0–2 map to video 0–2; indices 3–3801 map to video index +12.
Video indices 3–14 have no GT image. Matching images does not establish why they
were omitted or certify historical prediction filename provenance.

The reviewed propagation uses the verified sequential source, rather than the
historical block extraction. Source preparation preserves all 3,814 video frames.

- [Correspondence evidence](../../../results/recompute-20260909T145908Z-full-correspondence/verification.json)
- [Verified source](../../../results/verified-source-20260911T164528649441Z/verification.json)
- Tools: `audit/verify_correspondence.py`, `audit/prepare_verified_frames.py`.
- Diagnostic tools: `audit/check_alignment.py`, `audit/verify_extraction.py`.

Do not rename images to force alignment; join through the recorded correspondence.
