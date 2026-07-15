"""Multi-mode YOLO-seg labeling tool.

Stages (run from repo root):
  download.py       -> pull images from the Files API into data/images/
  run_chunk_seed.py -> optional classical/YOLO chunk polygon seeds
  app_multi.py      -> hand-label UI (http://127.0.0.1:5001)
  predict_batch.py  -> YOLO predictions for model-assisted review
  app_review.py     -> approve / reject / edit predictions
  export_multi.py   -> per-mode YOLO-seg dataset export
"""
