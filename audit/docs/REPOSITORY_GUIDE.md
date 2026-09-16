# Repository guide

A branch is a version of the project. A commit is a recorded set of changes;
pushing publishes those commits to GitHub and pulling brings them to another
checkout. The branch is not a separate folder for every experiment.

## Where to look

- Root README: application entry point and link to this research workflow.
- `audit/docs/tasks/`: current documentation grouped by task.
- `audit/docs/archive/`: superseded narrative investigations, clearly labelled.
- `audit/*.py`: executable tools and tests; retained paths preserve imports.
- `audit/results/`: timestamped evidence runs, indexed by its README.
- `audit/*CLOSEOUT.json`: machine-readable release records; filenames are stable.

Current findings should have one authoritative location. Old findings can be
archived with a superseded label rather than silently mixed with new conclusions.
Deleting an obsolete file is safe only after checking imports, command references,
provenance and reproducibility needs. Git retains prior committed versions, but
deletion still breaks current scripts that expect a path to exist.

This reorganisation moves prose, not input datasets or runnable scripts. Historical
run evidence and hash-pinned manifests remain unchanged, including original
wording embedded in frozen evidence. Current documentation uses neutral project
statuses. Actual timestamps identify recorded runs; future work is described by
its prerequisites rather than meeting dates.

For later code cleanup, first separate shared helpers from historical command
entry points, add compatibility checks, then retire unused entry points. That is
a code refactor rather than a documentation move and should be reviewed separately.

Do not commit checkpoints, raw videos, secrets or temporary environments. Local
scratch files under `audit/.tmp/` are ignored. Keep original data and released
results intact; corrected outputs should be separate, explicitly labelled runs.
