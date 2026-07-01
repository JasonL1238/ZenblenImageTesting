"""Standalone SAM-segmentation labeling tool.

Four stages, run in order:
  download.py  -> pull images from the Files API
  run_sam.py   -> run SAM to get candidate masks + polygons (needs the SAM conda env)
  app.py       -> Flask UI to accept/reject/correct each mask
  export.py    -> build the labeled training dataset

See labeling/README.md.
"""
