# Image gradient summaries

Run from the project root:

```powershell
.\.venv\Scripts\python.exe backend/extract_gradient.py "image.png"
```

Only three values are printed (ratios use 0–1): maximum window mean normalized
gradient, fraction of windows above 0.1, and union pixel coverage of those windows.
These summaries are project additions, **not paper-defined attack scores**.

LGS (https://arxiv.org/abs/1807.01216, equations 5, 6, 8) supplies the gradient
magnitude, global min-max normalization and window mean thresholding. The window
is 15×15, stride 10 (overlap 5), and selection uses strictly greater than 0.1.
We do not perform image smoothing, inference or train a classifier.

Implementation conventions not fully specified in the paper: Pillow luminance
grayscale, finite differences via numpy.gradient (central interior, one-sided
edges), original resolution, append an edge-aligned window when needed, use the
entire axis for images smaller than 15 pixels, and zero normalized response for
constant magnitude maps. These choices mean this is LGS-inspired extraction,
not a claim of exact author implementation reproduction. Normal text and texture
can have high responses; validate against normal documents before detection use.

The existing data/image_features/features.sqlite3 stores an independent gradient
record keyed by image SHA256 and gradient configuration version. Full window
means, coordinates and selected flags are in windows.npz under that image/config.
Completed results survive process restart; frequency records remain independent.
Original bytes are shared. No separate JSON export or model dependency is added.
