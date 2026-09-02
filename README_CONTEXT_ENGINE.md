# Taiwan Context Engine v0.1

This package adds a conservative decision layer after dual-pass ASR.

## Functions

- Rerank baseline and Taiwan-prompted ASR candidates.
- Add a small bonus when a candidate preserves Taiwan-context vocabulary.
- Detect low-confidence or strongly disagreeing candidates.
- Request user confirmation instead of silently fabricating a correction.
- Evaluate automatic coverage and CER using existing router JSON files.

## Install in the project

Copy the included `backend`, `datasets`, and `evaluation` folders into the
project root (`C:\taiwan-context-ai`). Existing files are not required to be
overwritten except when intentionally updating the same path.

## Evaluate

From the project root:

```bat
python evaluation\evaluate_context_engine.py evaluation\asr_validation\router_validation_margin_n005.json --output evaluation\asr_validation\context_engine_results.json
```

The engine does not use reference transcripts when making decisions. Reference
CER values in router JSON are read only after selection for offline evaluation.
