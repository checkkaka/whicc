# AGENTS.md

## Cursor Cloud specific instructions

**Platform reality:** `whicc` is a macOS 15+ / Apple Silicon desktop app. The full
product (SwiftUI `macui/`, MLX ASR in `src/whicc.py`, the `bin/audiotee` system-audio
capture binary, and `.app` packaging via xcodegen/Xcode) **cannot build or run on this
Linux cloud VM**. Full E2E — and the macOS build/i18n-lint CI in
`.github/workflows/ci.yml` — requires a real macOS runner. See `DEVELOPMENT.md` for the
macOS dev/build/packaging flow and `README.md` for the architecture.

**What runs on Linux here:** only the Python unit-test suite under `tests/`, which
exercises real production modules (`glossary_refresher`, `translate_stream`, `audio`,
`process_resolver`, `languages`).

**Dependencies:** `requirements.txt` pins the full macOS runtime (`mlx`, `mlx-metal`,
`mlx-audio`, `sounddevice`, …) and **will not `pip install` on Linux** (`mlx-metal` is
macOS-only). The startup update script instead creates a `venv/` and installs just the
cross-platform subset the tests need: `numpy`, `jieba`, `pytest`. This works because the
heavy/macOS-only deps (`mlx*`, `sounddevice`, `httpx`) are lazy-imported inside functions,
so the test modules import cleanly without them. `whicc.py` itself imports MLX at module
load and is not importable here.

**Run the tests** (the venv from the update script has no activate step needed):

```bash
PYTHONPATH=src ./venv/bin/python -m pytest tests/ -q
```

`PYTHONPATH=src` is required — the tests import top-level modules from `src/`.

**System note:** `python3-venv` is installed at the OS level (captured in the VM
snapshot), so the update script's `python3 -m venv venv` works without apt.
