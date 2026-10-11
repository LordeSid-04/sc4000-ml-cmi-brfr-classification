# Cleaning handoff verification

Verified locally on 11 October 2026 with Python 3.12.14, numpy 2.3.5 and pandas 3.0.1.

- Eleven automated test cases passed, including several invalid-input subcases. Coverage includes missing temperature channels and gaps, uneven counter spacing, invalid versus genuine zero distance readings, unavailable/extreme rotation, normalization, sorting, input nonmutation, stable channel ordering and independence from labels. CLI tests cover sequences crossing read-chunk boundaries, the 100-sequence output-part boundary, saved-output loading, empty-input rejection, rejecting overwrites, noncontiguous sequences and missing demographic subjects.
- The revised script was run from the original Kaggle ZIP to a new output directory. It completed all 574,945 rows and 8,151 sequences in 82 parts.
- Every saved part was reopened. Sensor arrays and masks matched in shape, all cleaned values were finite, metadata aligned with arrays, and every sequence index selected the correct sequence rows.
- All saved rotation vectors had unit length within float32 tolerance. This includes identity placeholders and does not imply unavailable orientations were recovered.
- Cleaned values and masks from the full rerun matched the previous complete dataset exactly.
- The function successfully cleaned both real example test sequences (56 and 51 rows) without gesture labels.
- Git ignore checks confirmed raw/generated datasets and local previews are excluded.
- All three Python code blocks in README.md were executed: saved-recording loading, valid-pixel-only distance summaries and prediction-time cleaning. CLI help was also checked.
- The final stricter CLI was rerun on the complete archive after adding contiguous-run validation, demographic subject checks and finite rotation-norm checks. It finished with `status: complete`; its values and observation masks were reconciled against the preceding full run.
- An additional independent source comparison checked 11 real recordings, selected to include missing rotation, temperature and distance. Original metadata and acceleration, independently calculated rotation normalization, numpy-based temperature filling, distance validity and the observation masks all matched saved outputs. The later empty-input guard does not change the transform for nonempty data; its CLI behavior was tested separately.

These checks establish reproducibility and the documented cleaning behavior on the supplied data. They do not establish that interpolation, rotation normalization or zero placeholders are optimal for any particular model. Model evaluation and correct use of masks remain necessary.

Local previews are in `preview/`: a compact first-recording view, all channels and flags for that recording, and real before/after examples. Local verification helper `inspect_cleaned_data.py` reruns saved-part reconciliation and creates those previews; it expects both completed output directories and the original audit results. It is not needed by the teammate to clean their data.
