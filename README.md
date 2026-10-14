# VN-AV-DF Forensics (v001 – v002)

**Detecting lip-sync deepfakes in Vietnamese videos with audio–mouth consistency and mechanism-based pseudo-fakes.**

This study is **closed** (14 October 2026). It consists of two versions run on the same pilot dataset, part1. The follow-up work, v003 (a Vietnamese lip-sync deepfake benchmark), is a separate project.

| | v001 (6 October 2026) | v002 (14 October 2026) |
|---|---|---|
| Question | Does an audio–mouth consistency branch help detect lip-sync fakes and generalise to unseen generators? | Can a detector trained **without any generator fakes** (only MechFake pseudo-fakes) generalise to unseen generators? |
| Proposed | **P2 SyncArtifact**: a sync branch (learned from reals, shams and shifted reals) + a DINOv2 artifact branch, with gated fusion | **MechFake**: 5 mechanisms faked from real videos (blending, downscale–upscale, VAE, warping, timing) + **TriCue**: 3 branches, 2 outputs ("AI-painted?" and "audio–visual mismatch?") |
| Baselines | B-FATE, **B-AVH** (AV-HuBERT + TCN), 5 P2 variants | B-AVH, P2, TriCue trained on Wav2Lip, 12 mechanism-analysis configurations |
| Detectors | 8 architectures × 3 seeds = 24 | 20 configurations × 3 seeds = 60 |
| Full report | `docs/v001_report.md` on branch [`archive/v001-part1-results`](../../tree/archive/v001-part1-results) | [docs/v002_report.md](docs/v002_report.md) (§15–§19) |

## Data (part1)

- **Source and clips:** 13 YouTube videos, **10 speakers**, 575 clean real clips (~1.2 hours). Clips are 5–8 s at 25 fps, short side ≤ 1080.
- **2×2 design:**
  - real;
  - sham (audio mismatch, no AI);
  - *source* fake (lips repainted to the original audio);
  - *donor* fake (lips repainted to other genuine speech of the same person).

  Partial fakes have random lengths.
- **Generators:** Wav2Lip (training) and MuseTalk 1.5 (**test only**, the unseen generator). v002 adds MechFake M1–M5 and two benign-processing diagnostics (label 0).
- **Splits:**
  - v001: 70/15/15, with 2 test speakers.
  - v002: re-exported 60/15/25, with **3 test speakers**; 7,892 samples / 16.4 hours.

## Main results (test, mean ± std over 3 seeds)

| | Overall AUC | Wav2Lip | **MuseTalk (unseen)** |
|---|---|---|---|
| v001 · B-AVH | 0.837 ± 0.025 | 0.889 | 0.786 |
| v001 · P2 | 0.766 ± 0.015 | 0.979 | 0.553 |
| v002 · B-AVH | 0.846 ± 0.008 | 0.892 | **0.800** |
| v002 · TriCue trained on Wav2Lip | 0.839 ± 0.019 | 0.990 | 0.689 |
| v002 · **TriCue-Mech** (no generator fakes seen) | 0.855 ± 0.035 | 0.933 | **0.776** |

The two versions use different splits, so **v001 and v002 numbers are not directly comparable**.

**Conclusions**
1. **On the unseen generator, training only on mechanism-based pseudo-fakes matches the supervised baseline:** Δ vs B-AVH = −0.015 [−0.061; +0.035]. It also **beats the same architecture trained on Wav2Lip** by +0.075 on MuseTalk. This holds for 2 of the 3 test speakers.
2. **MechFake models find the painted segment.** On 88% of partial MuseTalk clips, the highest-scoring cell falls inside the painted segment. Models trained on Wav2Lip reach only 27–42%.
3. **The visual-trace branch cannot yet separate "AI painting" from "processing".** It flags 79–100% of real clips whose mouth was merely pasted back. Only B-AVH passes this diagnostic.
4. **The audio–mouth consistency branch is useful for the "mismatch?" output** (sham 0.94; MuseTalk donor 0.83), not for the "AI-painted?" question.
5. **The main limitation is data:** **only 3 test speakers**, so no per-speaker confidence intervals yet.

## Repository layout

| Folder | Contents |
|---|---|
| `src/vn_av_df/` | Fake generation (`generation.py`, `mechfake.py`), features, models (`tricue.py`, `syncartifact.py`), training/testing (`experiment.py`), reporting |
| `data_pipeline/` | Collect → download → cut → review → export + split |
| `notebooks/` | `cut`, `generate`, `prepare`, `train` (Kaggle) and `compare` |
| `generation/`, `training/` | Local entry points |
| `docs/` | v002 report |
| `tests/` | Tests (fake generators, feature fixtures) |

How to run each step, local setup, Kaggle limits and checkpoint versions are covered in **[PIPELINE.md](PIPELINE.md)** (written in Vietnamese).

## Code versions

| Version | Git |
|---|---|
| v001 | branch `archive/v001-part1-results` (code at commit `6bfeaae`, report in `docs/v001_report.md`) |
| v002 | `main`, tag `v002` |

## Data, results and archives

Data, feature caches, checkpoints and results are **not stored in this repository**.
- They are kept in a private archive, one folder per version, each with a `README.md` and `SHA256SUMS.txt`:
  - `VN-AV-DF_v001-part1_2026-10-06`
  - `VN-AV-DF_v002-part1_2026-10-14`: code, reports, results of the 60 detectors, the generated dataset, AV-HuBERT + DINOv2 caches, the clean part1, and the raw data (private).
- The `v002` release on GitHub carries the results and reports, without videos.

## Ethics and usage limits

- Source videos are public YouTube videos. **Videos and fake samples of real people are not redistributed**; the data is for research only.
- Some generator checkpoints are restricted to non-commercial use (Wav2Lip: research / non-commercial).
- Detector scores are suspicion levels in 0–1 and are **not calibrated** probabilities.
