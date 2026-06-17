# Project: ZenblenImageTesting — smoothie blendedness CV pipeline

# Commands
- Test (single):    `pytest smoothie_cv/tests/test_pipeline.py::TestClassicalCVPipeline -v`
- Test (full):      `pytest smoothie_cv/tests/test_pipeline.py -v`
- Run classical:    `python run.py --pipeline classical --image <img.jpg>`
- Run VLM:          `ANTHROPIC_API_KEY=sk-... python run.py --pipeline vlm --image <img.jpg>`
- Run SAM:          `python run.py --pipeline sam --image <img.jpg>`
- Run all pipelines:`python run.py --pipeline all --image data/images/`
- Batch compare:    `python run.py --pipeline all --image data/images/ --threshold 0.90`

# Code style
- Type-hint all public function signatures
- No hardcoded API keys — always read from environment variables

# Workflow
- Prefer running a single test class over the full suite for speed.
- After CV edits: write result image to `outputs/`, then READ it back — never assert
  success without inspecting the actual mask overlay.
- Gate "done" on a numeric blend_score in [0, 1], not on the image looking right.
- Use a subagent for image analysis and multi-file investigation to keep context lean.

# Gotchas / environment
- VLM pipeline requires `ANTHROPIC_API_KEY` env var; missing key raises `EnvironmentError`.
- SAM pipeline requires checkpoint files in `checkpoints/` — download separately:
  `wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -P checkpoints/`
- SAM2 is installed from source, not PyPI: see `requirements.txt` comments.
- M4 Pro uses MPS backend (`torch.backends.mps`); Jetson Nano falls back to CPU — no CUDA.
- All pipelines share the same `BlendResult` contract — swap via `--pipeline` flag only.

# Session health (canary)
- Begin EVERY response with the marker 🟢 followed by a space.
- This is a context-health check — never skip it. If it starts disappearing,
  the session context is degrading: /clear (or /compact) and re-anchor.

# Compaction
- When compacting, always preserve: the list of modified files, the chosen
  approach and WHY, and any test / run commands.
