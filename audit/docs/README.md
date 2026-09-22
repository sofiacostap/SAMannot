# Task dashboard

| Task | Status | Documentation |
|---|---|---|
| 1 — Image correspondence | Verified source prepared; historical outputs have separate provenance limits | [Alignment](tasks/01-alignment/README.md) |
| 2 — GT reference | Screened reference released; exhaustive visual review pending | [Reference status](tasks/02-reference/STATUS.md), [visual review](tasks/02-reference/VISUAL_REVIEW.md) |
| 3 — Annotation and propagation | Reviewed baseline generated | [Workflow](tasks/03-annotation-propagation/WORKFLOW.md) |
| 4 — Quality scoring | Evaluation complete; scorer not deployment-ready | [Findings](tasks/04-scoring/FINDINGS.md), [protocol](tasks/04-scoring/PROTOCOL.md) |
| 5 — Identity review | Six proposals; correction pending visual approval | [Findings](tasks/05-identity-review/FINDINGS.md), [protocol](tasks/05-identity-review/PROTOCOL.md) |
| 6 — Temporal analysis | Baseline analysis complete; corrected comparison pending correction | [Findings](tasks/06-temporal/FINDINGS.md) |

Start with the [project overview](PROJECT_OVERVIEW.md). Use the
[script catalogue](SCRIPTS.md) to choose a tool and the [results catalogue](../results/README.md)
to choose its inputs. No propagation or evaluation rerun is required to read these findings.

Commands are run from the repository root. In task narratives, abbreviated
`results/...` paths refer to `audit/results/...`; clickable links resolve directly.

The analysis is descriptive evidence from one video with screened reference masks
and GT-assisted annotation. Completion of an evaluation does not establish perfect
GT or reliable deployment performance. The baseline is preserved separately from
any future human-assisted correction.

[Historical investigations](archive/README.md) explain previous bugs and provisional
results. They must not be combined with the released baseline statistics.
