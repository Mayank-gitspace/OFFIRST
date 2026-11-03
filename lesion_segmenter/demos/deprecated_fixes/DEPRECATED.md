# DEPRECATED (2026-11-02)
cache_probs.py / analyze.py / sweep_thr.py / combos.py / run_demo.py tuned min_score on IDRiD
Disease-Grading grade-0 images. That set is now evaluation-only and must not be used.
The min_score values they produced (MA .965, HE .95, EX .94, SE .965) are for the old hires1440
checkpoint only. Use lesion_detector/tune_thresholds.py (RFMiD-train normals + IDRiD seg test) instead.
