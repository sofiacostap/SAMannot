# Organisation review — 2026-09-16

The original audit README described the initial investigation rather than the
released pipeline. Task 2 mixed current release counts with superseded provisional
counts and commands. Task 4 described temporal analysis as pending after its
completion. Task 3 still described GPU integration as a future step. These are
documentation-state problems; they do not change the released measurements.

Current prose is grouped by task. Historical reports are labelled and archived.
The root README and audit README link to the task dashboard. A results catalogue
identifies the released evidence chain and separates earlier diagnostic runs.
Meeting deadlines and personal attribution have been removed from current prose
and the mutable Task 5 review status. Recorded timestamps still identify evidence.

Executable scripts, tests, frozen results and closeout records retain their paths.
Historical helpers are still imported by current code, so no executable was
deleted as obsolete. The script catalogue distinguishes supported workflow
entry points from supporting tools and superseded entry points. Further removal
requires extracting shared functions and checking compatibility first.

Verification: 65 local audit tests passed; grouped-document relative links were
checked. No GPU inference, project identity correction or result recomputation
was run. Unchanged frozen evidence may retain original narrative wording because
rewriting it would invalidate provenance checksums. This cleanup does not certify
all standalone commands against a different dataset or environment.
