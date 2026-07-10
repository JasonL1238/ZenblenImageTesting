# Test #2 — ORB/SIFT feature-matching to a canonical wordmark

**Verdict: does NOT work as a training-free clipped-wordmark fix. Confirms a trained YOLO logo class is needed.**

## Results on the 7 residuals
- **On-target (usable, non-degenerate exclusion box): 2/7.**
- 4/7 nominally "overlap" the leaked fragment, but 2 of those (4bf0d44c, adfcd90b)
  only overlap via a degenerate 190k / 82k-px homography box spilling off-frame —
  useless as an exclusion mask.
- Clean-cup false-positive rate: effectively **0**. 3/4 clean cups localized, but
  every hit lands on the *real printed wordmark*, not hallucinated smoothie. The
  failure mode is degenerate geometry, not inventing logos.

## Why it fails
- **ORB is dead:** only 4 keypoints on the low-contrast cream-on-pink reference →
  0 good matches on all 11 images. Eliminated.
- **SIFT reference:** 68 keypoints. Localizes only where the main wordmark is
  already clearly visible — exactly the case the existing text-line detector
  already catches. No new coverage.
- The genuinely hard residuals fail:
  - dce9974f — faded "len" on pale-pink → 1 match, no localization.
  - d71aa159 — 16 matches but a degenerate thin box shifted off-target.
  - 572b0f91 — correctly found "enblen" high on the cup, but the leak is the small
    "freshly blended" sub-text the wordmark reference can't contain → off-target.
- Curvature (letters wrap the cylinder) violates the planar-homography assumption;
  with only 5–16 ratio-passing matches, findHomography returns wildly stretched quads.

## Recommendation
Drop feature-matching for this problem. It only succeeds where the current logo
detector already succeeds, and fails on precisely the faded/curved/clipped fragments
and sub-text leaks that define the residual. A learned segmenter tolerates curvature,
partial occlusion, and low contrast and yields the tight per-pixel mask this cannot.

## Deliverables (this dir)
- reference_wordmark.png / reference_wordmark_enhanced.png — canonical crop (98c71591)
- results.csv — per-image ORB/SIFT keypoints, matches, inliers, localized/on_target/usable, box area
- overlays/<short>.png (11) — matched keypoints, localized box, true leaked-fragment bbox (green)
- run_experiment.py, _leaked_bboxes.json — ground-truth bboxes from the real pipeline (read-only)
