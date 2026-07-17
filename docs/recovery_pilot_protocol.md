# Recovery and Memory Development Pilot

The isolated Recovery/Memory pilot completed on the 20-task development set
from clean commit `34d0bb5`. It reused the same initial P0 plan across all six
harness modes, allowed at most one repair, and did not access the frozen
held-out set.

| Mode | Final success | Recovered initial failures | Conditional recovery | Paired uplift vs H0 | McNemar p |
|---|---:|---:|---:|---:|---:|
| H0 open loop | 4/20 (20%) | 0/16 | 0% | — | — |
| H2 local | 4/20 (20%) | 0/16 | 0% | 0 pp | 1.0000 |
| H2 LLM reflection | 12/20 (60%) | 8/16 | 50.0% | +40 pp | 0.0078 |
| H2 error-specific | 12/20 (60%) | 8/16 | 50.0% | +40 pp | 0.0078 |
| H2 memory | 11/20 (55%) | 7/16 | 43.8% | +35 pp | 0.0156 |
| H2 PDDL | 20/20 (100%) | 16/16 | 100% | +80 pp | 0.00003 |

Plain reflection and error-specific repair produced identical paired outcomes;
the error-specific variant also used slightly more attributed tokens. The
frozen symbolic-teacher memory improved over H0, but it did not improve over
plain reflection and used more tokens. Local repair did not solve any standard
planning failure in this pilot, although it remains useful in the separate
frozen safety benchmark. PDDL recovery is reported only as a symbolic recovery
reference.

The combined harness and leave-one-component-out pilot will not be run. The
isolated results do not justify paying for a combination whose local component
showed no gain, whose error-specific component duplicated reflection, and whose
memory component was slightly worse than reflection.

The final confirmatory recovery comparison is therefore frozen to P1 with H0,
plain LLM reflection, symbolic-teacher memory, and PDDL recovery. Error-specific
and local results remain development ablations and must still be reported.

Machine-readable evidence is stored in `docs/recovery_pilot_evidence.json`.
The source run is
`runs/pilot_recovery_deepseek_20/20260717T043432879063Z_9d8b5ba3`.
