# Highlight Selection Audit (2026-11-16)

## Scope and evidence

Inspected `app.py`, `config.py`, `modules/highlight_scoring.py`, `modules/event_classification.py`, `modules/object_detection.py`, `modules/audio_analysis.py`, `modules/speech_recognition.py`, `modules/clip_editor.py`, `modules/title_generator.py`, `modules/description_generator.py`, the current test suite, project progress notes, staged media, and existing output metadata.

The only source video currently in `downloads/` is `Grand Theft Auto VI Trailer 1.mp4`. `input/` contains two tiny text files and no gameplay recording. Existing exported clips point to a now-missing extended-look trailer path. Their metadata includes confident chase/heist/crash claims and generated descriptions despite the available signals not supporting those labels. The progress document asserts gameplay event performance, but the visible tests exercise synthetic signals; there is no evidence in this workspace of labeled gameplay evaluation.

## Existing pipeline

1. `modules/audio_analysis.py` turns RMS and onset strength into one-second buckets. Relative-to-file 99th percentile thresholds label loud transients as gunshots/explosions; these are not sound-classification models and cannot distinguish GTA sound events reliably.
2. `modules/object_detection.py` samples one frame per second. COCO YOLO detections are converted into a sum of class weights, frame difference is used as motion, and a mean pixel difference over 35 becomes a scene cut. No object tracking, optical flow, visual description, or event model is used.
3. `modules/event_classification.py` labels each second independently. Any audio transient plus motion can become a `police_chase`; high motion plus generic visual detections can become a `crash` or `stunt`. There is no police-vehicle evidence, impact evidence, pursuit progression, or temporal outcome check. `vehicle_theft` is configured but never detected.
4. `modules/highlight_scoring.py` independently min-max normalizes each signal across the whole video. It combines YOLO object counts, motion, loudness, onset spikes, heuristic event bonuses, speech density, and onset spikes again as payoff potential. Thus loudness/spikes are counted twice, while every signal gets an artificially high relative score even in uniformly boring footage. Event bonus normalization also makes the largest heuristic label dominate its own factor.
5. Candidate windows are generated exhaustively at 1/2-second start increments for only min/mid/max durations. They are ranked by average second score, not by event progression, clarity, build-up, or payoff. The single peak second is retained; there is no event-centered boundary detection.
6. Greedy non-overlap blocks literal overlap, but does not compare event identity or similarity. Long candidate windows can contain several events and obscure duplicates; payoff can be cut at the window edge. The event category is the most frequent per-second primary label (usually `unknown`), not a grounded multi-label result.
7. `app.py` immediately generates metadata and exports every selected clip in the same run. There is no pre-export contact sheet, candidate preview, approval/rejection step, timestamp editor, or low-confidence warning.
8. Metadata generators receive heuristic event labels and transcript only, not sampled clip frames. Template hooks are generic and sometimes assert unsupported content (for example, a police chase); fallback descriptions likewise assert a clip moment without visual evidence.

## Root causes

- Weak proxies are presented as semantic GTA event understanding.
- Per-second independent labels cannot represent setup → action → consequence.
- Global min-max normalization rewards relative outliers even if the entire recording is dull.
- YOLO detections and audio spikes are treated as positive evidence without corroboration, and spikes are double-counted.
- Fixed-duration average windows dilute short moments and may miss the payoff.
- The application has no human review gate before exporting or generating publishable metadata.
- Validation is synthetic/unit-level rather than human-labeled gameplay, and existing output metadata is not trustworthy evidence of quality.

## Safe implementation direction

First improve confidence calibration and temporal grouping using the current local signals, explicitly marking heuristic categories as tentative and requiring corroboration. Add explainable factor breakdowns, event-focused window construction, deduplication, a contact sheet/preview review gate, and approval-only export. Avoid claiming event-model accuracy or measured quality gains until real gameplay plus human labels are supplied. The RTX 3050 has a 6 GB reported capacity, but this checkout has no benchmark evidence justifying an additional vision-language or audio-classification model; do not add one based on guesswork.

## Changes made after the audit

- Replaced per-file min/max scaling with fixed, bounded signal scaling and configurable factor weights. Audio contributes 10% of the default factor weights; transients are not counted again as payoff.
- Added an explicit low-motion/low-evidence penalty and smoothed sustained activity regions. Candidate windows add short context around an active region, respect the selected duration range, and report window mean, activity coverage, event confidence, and a reason.
- Replaced unsupported semantic guesses with weak evidence tags (`possible_audio_transient`, `sustained_motion`, `speech`, `scene_change`) and `unknown` as the category. The current stack cannot substantiate GTA-specific chase/crash/stunt/funny labels.
- The first implementation added a review gate, which made the app stop before exporting. That was a mistake and has been removed: the main button now runs detection, picks windows, cuts clips, and burns captions in the same run. The user does not enter timestamps.
- Replaced activity-region-only windowing with a sliding search over the user's duration range. The ranker scores mean activity, peak activity, sustained coverage, and baseline lift, then rejects overlapping windows.
- Fixed duration selection after discovering that the first formula still favored the minimum. It now estimates the full sustained activity span around each peak, adds context, and ranks durations across the configured range separately from clip count.
- Updated synthetic tests to assert that motion plus an audio transient does not become a named GTA event, and that 15 distinct events produce 15 clips with varied durations inside 15–120 seconds. All nine test suites passed after setting temporary-file paths inside the writable workspace.

## Still unverified / incomplete

No gameplay file or human-selected timestamps are available in this workspace. Therefore precision, duplicate-event rate, payoff accuracy, processing-time comparison, and VRAM use on gameplay are not measured. The current one-frame-per-second YOLO and audio heuristics still cannot understand event semantics or natural endings; the boundary padding is a heuristic. The review UI is the safety net until gameplay examples can ground further tuning. No additional model was installed or benchmarked.
