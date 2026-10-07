# Datasheet for BATON

Following Gebru et al., "Datasheets for Datasets" (CACM 2021).

## Motivation
**Purpose.** BATON supports research on bidirectional driver–automation control transitions (engagement and disengagement of Level-2 driver assistance) in naturalistic driving, including handover prediction, takeover onset detection, and pre-override takeover anticipation.
**Creators.** The BATON team (University of South Florida and collaborators).
**Funding.** Academic research; no commercial sponsor influenced collection or labeling.

## Composition
- Full corpus: 781 route bundles, 204.9 driving hours, 173 unique drivers, 108 vehicle-model fingerprints across 22 manufacturers.
- Frozen benchmark subset (`benchmark_v2`): 565 route bundles, 162.1 h, 150 unique drivers, 99 vehicle models, 3,593 control transitions (1,800 engagements / 1,793 disengagements).
- Modalities per route: front-view video (20 fps), in-cabin video (20 fps), CAN-decoded vehicle dynamics (100 Hz), IMU (100 Hz), forward radar (20 Hz), driver-monitoring outputs (20 Hz), planner outputs (20 Hz), GNSS-derived route context (raw coordinates withheld).
- **Identity semantics.** Each driver has a unique, pseudonymous driver ID; one recording setup is installed per vehicle and used by a single driver, so driver IDs correspond one-to-one to drivers. 4/150 benchmark drivers recorded in more than one vehicle model; 28/99 models are shared by ≥2 drivers.
- **Known skews.** Top 5 drivers account for 31.3% of benchmark hours (top 20: 60.1%); 80 drivers contribute <30 min; top 5 vehicle models cover 40.8% of hours. Software spans openpilot 0.7–0.11.x and comma-release builds on three recording-hardware generations (tizi/tici/mici). Engagement structure: both-axes 38.5% of driving time, lateral-only 9.5%, longitudinal-only 6.8%, manual 45.2%.
- **What is not covered.** No demographic attributes, no administrative region, no weather labels, no physiological or eye-tracking signals. Fairness across demographic groups is not evaluable from the released data.

## Collection Process
- Recording hardware: comma devices (comma three / 3X) mounted at the windshield center; CAN accessed via the car harness; openpilot logging stack timestamps all streams on a common route-level clock.
- Recruitment: collection began with five core drivers in Tampa, Florida, and expanded through direct collaboration, contributor outreach, and permission-based access to shared recordings from the comma/openpilot ecosystem. Contributors are self-selected driver-assistance users; the corpus therefore over-represents ADAS enthusiasts and their vehicles, routes, and regions.
- Consent basis (uniform across the corpus): explicit permission for research use obtained through direct communication with each contributor, under comma's publicly posted Terms and Privacy Policy. Most routes had already been publicly shared by their owners on the platform, but public sharing is never treated as research consent. The team's contribution is communication, verification, curation, and standardization. Secondary use of platform-collected data; not conducted under an IRB protocol.

## Preprocessing and Labeling
- CAN decoded with public OpenDBC definitions and the OpenLKA cross-vehicle pipeline; structured streams resampled to 50 Hz for the benchmark loader.
- Transitions defined from lateral/longitudinal assistance flags (OR-coarsened; subtype flags released) with debounce and stability filters (see paper Appendix D).
- Task-1 action labels are rule-derived (single-label priority variant and multi-label variant both released) and validated by threshold-sensitivity analysis and blind human annotation; a second-annotator kit (`annotation_kit_v2/`) supports inter-rater measurement.
- Leak-safe and anticipation-safe input protocols withhold automation-control and driver-override channels respectively.

## Uses
- Intended: benchmarking transition prediction/detection/anticipation, multimodal fusion, driver-state modeling, leakage-aware evaluation methodology.
- **When not to treat BATON as representative:** population-level claims about drivers in general; region-, weather-, or demographic-conditioned analyses; per-driver behavioral profiling of identifiable individuals (prohibited); safety certification of production systems.
- Prohibited: re-identification of drivers, vehicles, or locations; surveillance; scoring identifiable individuals.

## Distribution
Three tiers: (1) unrestricted public benchmark package — benchmark-ready 50 Hz tensors, video/pose features (in-cabin content only as non-invertible visual features), sample CSVs, splits, labels, code (GitHub `OpenLKA/BATON`, HuggingFace `HenryYHW/BATON-Sample`); (2) traceable request form for server-produced custom raw-stream views in any format; (3) privacy-sensitive raw tier — the complete raw corpus including driver-facing source data (HuggingFace `HenryYHW/BATON`), released in full upon a clear identity-confirmation form; verification is the only step, no research purpose is screened, and the public benchmark tier alone suffices to reproduce every reported result. License: CC BY-NC 4.0 (data), MIT (code). Checksums in `benchmark_v2/CHECKSUMS.sha256`; reproduction guide in `REPRODUCE.md`.

## Maintenance
Versioned, additive releases; published splits never change; errata ship as versioned patches with changelogs; contributor removal requests propagate to all tiers and are recorded in the next version's changelog. Contact via the GitHub repository issue tracker.
