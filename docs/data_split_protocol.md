# Frozen Data Split Protocol

## Roles

| Artifact | Tasks | Role |
|---|---:|---|
| `rag_train.jsonl` | 75 | RAG examples and training-only resources |
| `balanced_eval.jsonl` | 202 | Development, debugging, ablations, and method selection |
| `heldout_virtualhome_119.jsonl` | 119 | One-shot local VirtualHome-only final evaluation |
| Official hidden test | Organizer controlled | Primary final benchmark |

The 20-task and 50-task pilots are subsets of the 202-task development set.
They are not separate test sets.

## Frozen Local Held-Out

The local held-out split is the set difference between the 321 executable
evaluation tasks and the 202 development tasks. The split contains 119 tasks,
all from VirtualHome:

- 77 easy
- 39 medium
- 3 hard
- 0 BEHAVIOR

Frozen identifiers and fingerprints:

- Task-ID SHA-256: `036ed8d9c943477bdc704d4d1e4fd3e84541352f8a132984b68c3b6c51f22eac`
- Task-record SHA-256: `4a01bdb4ce499fdc20fc2d11ad4b4139296bf5a9b936525bc730d214776b3bb9`
- Development/held-out overlap: 0

Authoritative artifacts:

- `data/processed/tasksets/heldout_virtualhome_119.jsonl`
- `data/processed/tasksets/heldout_virtualhome_119_ids.json`
- `data/processed/tasksets/heldout_virtualhome_119_manifest.json`

## Non-Leakage Rules

1. Never inspect held-out per-task results during method development.
2. Never add prompts, macros, repair rules, RAG examples, or memory entries in
   response to a held-out failure.
3. Build and select methods only on training and the 202-task development set.
4. Freeze code commit, prompts, RAG corpus, memory, models, and statistics before
   the first held-out run.
5. Run the local held-out evaluation once for the preregistered final cells.
6. Report the local result as VirtualHome-only; it is not an unbiased BEHAVIOR
   estimate and is not an official challenge score.

The official hidden test is required for the primary cross-environment final
claim because no untouched local BEHAVIOR split remains.

## Reproduction Guard

The freezer refuses count changes, dataset changes, development tasks missing
from the executable inventory, duplicate IDs, or changes to existing frozen
artifacts:

```bash
uv run embodied-gap freeze-heldout \
  --executable data/processed/tasksets/executable_eval.jsonl \
  --development data/processed/tasksets/balanced_eval.jsonl \
  --out-dir data/processed/tasksets \
  --name heldout_virtualhome_119 \
  --expected-count 119 \
  --expected-dataset virtualhome
```
