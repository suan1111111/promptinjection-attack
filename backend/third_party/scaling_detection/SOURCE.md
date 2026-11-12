# Source attribution

Repository: https://github.com/EQuiw/2023-detection-scalingattacks
Publication: Quiring, Müller and Rieck, ACSAC 2023.
Retrieved 2026-10-10. f_utils.py is copied unchanged from:
scaleatt/defenses/detection/frequency/spectrum/utils/f_utils.py
License: GPL-3.0, retained in LICENSE.

Adapted scoring in services/scaling_features.py follows PeakSamplingStrategy.strategy
and ScoringStrategy.get_percentile_of_score. Down/up follows DownAndUpScalingDefense
and scale_pillow; PSNR replaces the obsolete skimage.measure.compare_psnr call
with its mathematical formula on RGB8 (data range 255).
The application wrapper, caching, CLI, and no-peak handling are project additions.
