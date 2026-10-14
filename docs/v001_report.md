# v001 — Vietnamese lip-sync deepfake detection with temporal audio–mouth consistency

Updated **6 October 2026**: full results on **part1** (24 detectors = 8 architectures × 3 seeds, trained and tested once). Descriptions follow the code as of v001 (this branch).

> **v001 is closed (14 October 2026).** The results and conclusions below are unchanged. The follow-up was completed as v002 (MechFake + TriCue; see `docs/v002_report.md` on `main`, §15–§19). Three caveats:
> - **The code on `main` differs from v001**: B-FATE, `avh_realrecon` and the 4 P2 variants were removed; MechFake/TriCue were added.
> - **The part1 split was later re-exported as 60/15/25** (3 test speakers). v001's 2-speaker test set survives only in the v001 archive and on this branch.
> - Therefore **do not compare v001 numbers directly** with v002 numbers.
>
> The status of v001's open items is in §10.3.

- **Part I — Report:** results, discussion, cost, open issues, next steps (v002).
- **Part II — Technical details:** data, labels, features, architectures, training, evaluation; can be checked against the code. References in §22.

---

# PART I — REPORT

## 1. Topic and scope

**Title:** *Vietnamese Audio-Visual Deepfake Detection Using Cross-Modal Temporal Consistency.*

**Direction (option A, 30/09/2026):**
- Temporal audio–mouth consistency is the **main axis** of the proposed method.
- The question "does the model learn visual artifacts or audio–mouth consistency?" is the **analysis part**, answered with 2×2 data and per-branch reporting.

**Flow:** YouTube video → cut real clips + manual review → **split at export** → generate fakes (Wav2Lip, MuseTalk) + shams → review/finalise → extract features → train 8 architectures → test → per-part comparison → demo.

**v1 scope:** one speaker, clearly visible face, with audio. The forgery is an **AI-repainted mouth** (lip-sync). The audio is always a genuine recording (the original, or the same person's speech from another segment).

**Detector outputs:**
- a whole-video suspicion score (0–1, **not calibrated**);
- suspicious intervals and coverage;
- for P2, additionally per-branch scores and timelines (sync and artifact).

## 2. Research questions and hypotheses

| Question | Experiment | Interpretation limit |
|---|---|---|
| **RQ1.** How well can Vietnamese lip-sync fakes be detected and localised? | 8 architectures, same dataset/protocol; AUC, AP, temporal AP | Within the data and the 2 generators tested |
| **RQ2.** Does an audio–mouth consistency branch (trained **without seeing fakes**) contribute beyond visual artifacts? | P2 vs `p2_artifact_only`, `p2_sync_only`, `p2_concat`; P2 vs B-AVH | P2 adds DINOv2, so beating B-AVH is not enough to credit consistency |
| **RQ3.** Does the model learn artifacts or consistency? | `by_condition` + `branches`: fake source (artifact only) vs donor (artifact + replaced audio); sham (mismatch, no AI) | Sham is only one kind of mismatch (audio splicing) |
| **RQ4.** What about an unseen generator (MuseTalk); which branch is more robust? | Wav2Lip (seen) vs MuseTalk (test only) | One held-out generator does not represent all deepfakes |
| **RQ5.** Does decoupling sync from fakes (freezing) matter? | P2 vs `p2_sync_seen_fake` | Needs ≥ 3 seeds |

Expected pattern:
- real: low on both branches;
- sham: high on sync, low on artifact (AI = 0);
- fake source: low on sync (if the generator is good), high on artifact;
- fake donor: high on both.

## 3. Data — 2×2 design

| | Original audio | Replaced/spliced audio (genuine speech of the same person) |
|---|---|---|
| **Original visual** | Real (AI=0) | Sham (AI=0, known A/V mismatch interval) |
| **AI visual** | Fake source (AI=1) | Fake donor (AI=1) |

| Cell | Train | Validation | Test |
|---|---|---|---|
| Real, Sham | Yes | Yes | Yes |
| Wav2Lip donor full/partial | Yes | Yes | Yes (seen generator) |
| Wav2Lip source full/partial | No | Yes | Yes |
| MuseTalk 1.5 donor/source full/partial | No | No | Yes (unseen generator) |

- Training uses **donor fakes only**: the model does not learn "artifact only" from source fakes; source is used only for measurement on val/test.
- **The split happens at step 05 export** (since 06/10/2026; previously at the generation plan step).
  - It is grouped by connected person/source/hash groups.
  - The ratio is a provisional **70/15/15** (80/10/10 with 10 speakers would leave only 1 validation speaker), and at least 3 groups are required.
  - The existing part1 was split with exactly this algorithm and seed, so re-exporting gives identical results.
- Parents with no same-`speaker_id` donor in the same split are dropped (`skipped_without_donor`).

## 4. Compared models (`ARCHITECTURES = "all"`)

| Code | Role | Idea |
|---|---|---|
| **B-FATE** `fate_gru` | Baseline 1 | PE-AV Small [1] + FATE adapter [2] (frozen) → BiGRU |
| **B-AVH** `avh_tcn` | Baseline 2 | AV-HuBERT [3] A/V (frozen) → TCN [20]; same backbone as P2, no consistency modelling |
| **P2** `p2_syncartifact` | Proposed | **Sync** branch (A→V reconstruction residual learned from real; the sync head learns real/sham/shifted real, **without seeing fakes**, then frozen) + **artifact** branch (DINOv2 [5] mouth crop) + gated fusion; the A→V residual is inspired by AuViRe [18] |
| `p2_artifact_only`, `p2_sync_only` | P2 variants | One branch only |
| `p2_concat` | P2 variant | Concatenate both branches, no gate |
| `p2_sync_seen_fake` | P2 variant | Sync branch not frozen in stage C (also learns from fakes) |
| `avh_realrecon` | AVH variant | A→V residual + visual, gate (old P1), no stage S, no DINOv2 |

Older architectures (`fate_linear`, `fate_tcn`, `fate_transformer`, `avh_linear`, `avh_visual`, `avh_concat`) were **removed from the code** on 06/10/2026.

## 5. Evaluation protocol

| Item | Rule |
|---|---|
| Seeds | 42/43/44 |
| Budget | 20 epochs, patience 5 for detectors; stage A and S 20 epochs / patience 5 each |
| Checkpoint / threshold selection | Lowest validation loss; video and temporal thresholds chosen on validation (max F1), kept fixed at test |
| Held-out | MuseTalk never enters train/validation |
| Test | **Run once right after training** (`RUN_TEST = True`), locked by `test-lock.json` |
| Comparison | `notebooks/compare.ipynb`: **per part**, mean ± std, seed ensemble, by generator / 2×2 cell / branch, temporal AP, **paired ΔAUC** with bootstrap over source clips |
| Main metrics | ROC-AUC, video AP, F1 / false alarm; temporal AP@0.3/0.5/0.75/0.95 |

## 6. Status (06/10/2026)

| Item | Status |
|---|---|
| Data | **Part1 done:** 575 clean clips / 10 speakers / 13 source videos (1.20 h real); 2954 generated samples (6.08 h) |
| Experiments | **24 detectors** (8 architectures × 3 seeds) trained and tested on part1; merged in `runs/train_part1_run01`; comparison tables in `reports/compare/` |
| Pipeline | 5 standardised Kaggle notebooks (cut, generate, prepare, train, compare); split at export; one generator per session; generated samples default to keep; `kaggle_pull.py` tool |
| Code | `tests/` + `data_pipeline/tests`: 106 passed, 1 skipped (symlinks on Windows); ruff clean |
| Demo | Builds; not yet run locally with real checkpoints |
| Git | Most of the new structure not yet committed |

## 7. Part1 results

### 7.1. Experiment record

| Field | Value |
|---|---|
| Run dates | Generation 05/10; prepare + training of the 3 main architectures 05/10; the 5 variants 06/10 |
| RUN_NAME / part / manifest | `train_part1_run01` / `vn-av-df-data-part1` / selection `631ea60e…`, part `2f3e43ee…` |
| Models and seeds | 8 architectures × {42, 43, 44}, all 24 |
| Configuration | Defaults (§5, §15); `CLIPS_PER_SPLIT = 0`, CRF 18, `max_side` 640 |
| Hardware | Kaggle 2×T4 (16 GB); AV-HuBERT worker: Python 3.10.21, torch 2.0.1+cu118 |
| Generators | Wav2Lip [6] `bac9a81e…` (ckpt `180cfd49…`), MuseTalk 1.5 [7] `0a89dec4…` (ckpt `7ebf6c98…`) |
| Encoders | PE-AV Small [1] `dd050762…`, FATE adapter [2] `8463ab93…`, FATE source `beae95ae…`; AV-HuBERT [3] Base LRS3 [4] iter4 (`a6476324…`); DINOv2-small [5] `ed25f3a3…` |

### 7.2. Data

| Step | Numbers |
|---|---|
| Sources | 15 rows in `videos.csv` → **13 videos** selected (2 rejected for quality) → **10 speakers** |
| Cutting | 852 candidate clips (106.8 min); 100 windows rejected while cutting |
| Real review | **575 keep** (71.8 min), 222 reject, 55 uncertain (67% kept) |
| Clips per speaker | 15–158 (speaker_09: 158, speaker_02: 15) — **strongly imbalanced** |
| Generation | 910 planned pairs; **4 pairs dropped** (Wav2Lip found no face, all in train: speaker_06 ×3, speaker_09 ×1); 0 parents without a donor |
| Fake review | 2954/2954 keep; partial lengths evenly spread over 0.4/0.8/1.6/2.4 s (221–231 samples each) |

| Split | Speakers | Clean clips (min) | Samples | Composition | Hours |
|---|---|---|---|---|---|
| Train | 5 (05, 06, 07, 09, 10) | 404 (51.4) | 1600 | 400 × {real, sham, W2L donor full, partial} | 3.40 |
| Validation | 3 (02, 03, 11) | 89 (10.7) | 534 | 89 × {real, sham, W2L donor/source × full/partial} | 1.07 |
| Test | **2 (01, 08)** | 82 (9.7) | 820 | 82 × {real, sham, (W2L, MT) × (donor, source) × (full, partial)} | 1.61 |

- Actual clean-clip ratio is 70.3/15.5/14.3.
- No speaker or source crosses two splits (connected groups).
- **The test set has only 2 speakers**: the per-source-clip bootstrap CIs in §7.7 are optimistic, and conclusions about new speakers are not yet supported.

### 7.3. P2 stages A/S

| Seed | R: val loss / constant | `1 − R/const` | Sync: val loss / AUC (real vs sham + shifted) | Time A / S |
|---|---|---|---|---|
| 42 | 0.4175 / 0.4147 | −0.007 | 0.497 / 0.784 | 88 s / 103 s |
| 43 | 0.4166 / 0.4147 | −0.005 | 0.524 / 0.780 | 76 s / 132 s |
| 44 | 0.4192 / 0.4147 | −0.011 | 0.441 / 0.777 | 68 s / 117 s |

**R (A→V reconstruction) is no better than a constant predictor in all 3 seeds** (early stopping after 6–8 epochs).
- By the rule set in advance, the A→V residual therefore has **no foundation** to be interpreted as "audio–mouth consistency".
- The sync head still separates real from sham/shifted audio (validation AUC ~0.78), because R(A) still changes with the audio. The sync branch reacts to **changes on the audio side**; it has not been shown to model the audio–mouth relationship.
- `avh_realrecon` uses the same R and gets the same result (0.4166–0.4192 vs 0.4147).

### 7.4. Video-level detection — test

| Model | ROC-AUC | Seed-ensemble AUC | Video AP | F1 | False alarm (real + sham) |
|---|---|---|---|---|---|
| B-FATE | 0.694 ± 0.002 | 0.699 | 0.909 ± 0.001 | 0.813 ± 0.031 | 0.612 ± 0.083 |
| **B-AVH** | **0.837 ± 0.025** | **0.849** | 0.955 ± 0.009 | 0.831 ± 0.012 | 0.242 ± 0.049 |
| AVH-realrecon | 0.824 ± 0.005 | 0.840 | 0.948 ± 0.002 | 0.869 ± 0.035 | 0.439 ± 0.165 |
| P2 | 0.766 ± 0.015 | 0.765 | 0.940 ± 0.005 | 0.758 ± 0.013 | 0.191 ± 0.035 |
| P2-concat | 0.771 ± 0.008 | 0.781 | 0.940 ± 0.003 | 0.735 ± 0.033 | 0.128 ± 0.063 |
| P2-sync-seen-fake | 0.790 ± 0.004 | 0.793 | 0.945 ± 0.002 | 0.744 ± 0.010 | **0.114 ± 0.025** |
| P2-sync-only | 0.550 ± 0.030 | 0.550 | 0.827 ± 0.021 | 0.871 ± 0.011 | 0.878 ± 0.042 |
| P2-artifact-only | 0.783 ± 0.026 | 0.785 | 0.944 ± 0.006 | 0.750 ± 0.009 | 0.134 ± 0.042 |

Video coverage is 100% for every model. Video AP is high because the test set has 656 fakes vs 164 label-0 clips (80% positives); ROC-AUC and false alarm are the main readings.

### 7.5. By generator and by 2×2 cell

| Model | MuseTalk (held-out) | Wav2Lip | W2L donor | W2L source | MT donor | MT source | FAR real | FAR sham |
|---|---|---|---|---|---|---|---|---|
| B-FATE | 0.494 | 0.895 | 0.890 | 0.905 | 0.504 | 0.498 | 0.606 | 0.618 |
| **B-AVH** | **0.786** | 0.889 | 0.938 | 0.844 | **0.825** | **0.757** | 0.248 | 0.236 |
| AVH-realrecon | 0.778 | 0.869 | 0.914 | 0.825 | 0.801 | 0.756 | 0.439 | 0.439 |
| P2 | 0.553 | **0.979** | 0.978 | 0.982 | 0.562 | 0.558 | 0.191 | 0.191 |
| P2-concat | 0.578 | 0.963 | 0.964 | 0.964 | 0.591 | 0.585 | 0.126 | 0.130 |
| P2-sync-seen-fake | 0.606 | 0.974 | 0.972 | 0.976 | 0.611 | 0.602 | 0.114 | 0.114 |
| P2-sync-only | 0.552 | 0.549 | 0.600 | 0.436 | 0.520 | 0.514 | 0.878 | 0.878 |
| P2-artifact-only | 0.589 | 0.977 | 0.973 | 0.980 | 0.592 | 0.587 | 0.134 | 0.134 |

Generator columns: that generator's fakes vs all label-0 clips. Cell columns: that cell's fakes vs original reals (mean over seeds; std in `reports/compare`).

### 7.6. By branch (threshold-free AUC, cell vs original real)

| Branch | Sham | W2L donor | W2L source | MT donor | MT source |
|---|---|---|---|---|---|
| P2 / sync | **0.884** | 0.677 | **0.436** | **0.803** | 0.739 |
| P2 / artifact | 0.500 | **0.980** | **0.984** | 0.643 | 0.635 |
| P2-sync-seen-fake / sync | 0.866 | 0.688 | 0.449 | 0.802 | 0.732 |
| P2-sync-seen-fake / artifact | 0.500 | 0.973 | 0.977 | 0.686 | 0.668 |
| P2-artifact-only / artifact | 0.500 | 0.978 | 0.983 | 0.666 | 0.649 |
| P2 fused | — (FAR 0.191) | 0.978 | 0.982 | 0.562 | 0.558 |

### 7.7. Paired ΔAUC (seed ensemble, 2000 bootstrap resamples over 82 source clips, 95% CI)

| Comparison | Overall | Wav2Lip | MuseTalk |
|---|---|---|---|
| P2 − B-AVH | **−0.083** [−0.113; −0.055] | **+0.088** [+0.067; +0.110] | **−0.255** [−0.298; −0.209] |
| P2 − B-FATE | +0.066 [+0.043; +0.092] | +0.084 [+0.059; +0.110] | +0.048 [+0.015; +0.084] |
| B-AVH − B-FATE | +0.150 [+0.121; +0.177] | −0.005 [−0.036; +0.026] | **+0.304** [+0.265; +0.344] |
| P2 − P2-artifact-only | −0.020 [−0.035; −0.010] | +0.003 [−0.002; +0.009] | −0.044 [−0.071; −0.024] |
| P2 − P2-concat | −0.016 [−0.028; −0.006] | +0.006 [+0.001; +0.011] | −0.037 [−0.057; −0.018] |
| P2 − P2-sync-seen-fake | −0.028 [−0.044; −0.014] | +0.007 [−0.001; +0.016] | −0.063 [−0.095; −0.041] |
| P2-sync-only − B-AVH | −0.298 [−0.350; −0.249] | −0.346 [−0.396; −0.297] | −0.250 [−0.309; −0.192] |
| AVH-realrecon − B-AVH | −0.009 [−0.028; +0.010] | −0.016 [−0.032; +0.001] | −0.002 [−0.025; +0.027] |

### 7.8. Temporal localisation (test, 820 videos with temporal labels)

| Model | AP@0.3 | AP@0.5 | AP@0.75 | AP@0.95 | Short recall@0.5 (≤1 s) | Start/end MAE @0.5 (s) | False intervals / hour of label 0 |
|---|---|---|---|---|---|---|---|
| B-FATE | 0.441 | 0.341 | 0.263 | 0.000 | 0.05 | 0.64 / 0.52 | ~432 |
| B-AVH | 0.457 | 0.338 | 0.194 | 0.11 | 0.17 | 0.44 / 0.34 | ~129 |
| AVH-realrecon | 0.427 | 0.330 | 0.179 | 0.10 | 0.17 | 0.43 / 0.41 | ~212 |
| **P2** | **0.488** | **0.420** | **0.316** | **0.20** | **0.44** | **0.21 / 0.20** | ~138 |
| P2-concat | 0.361 | 0.289 | 0.209 | 0.13 | 0.39 | 0.20 / 0.27 | ~99 |
| P2-sync-seen-fake | 0.441 | 0.370 | 0.255 | 0.15 | 0.39 | 0.25 / 0.30 | ~80 |
| P2-sync-only | 0.338 | 0.193 | 0.157 | 0.14 | 0.03 | — | ~724 |
| P2-artifact-only | 0.433 | 0.352 | 0.245 | 0.15 | 0.37 | 0.27 / 0.29 | ~85 |

AP comes from an in-house evaluator (§16) and is not directly comparable with numbers in papers. MAE is computed on matched intervals only.

### 7.9. Cost and GPU projections

**Measured (part1, 2954 samples, Kaggle 2×T4):**

| Item | Time |
|---|---|
| FATE extraction | 5.57 h (2 GPUs, `batch_windows` 8) |
| AV-HuBERT extraction | 4.73 h (2 GPUs × 2 workers; dlib on CPU is the bottleneck) |
| DINOv2 extraction | 0.40 h |
| Training 24 detectors | 2.78 h total; per seed: B-FATE 11.2 min, B-AVH 3.5, AVH-realrecon 5.5, P2 5.8 (stage A ~80 s + S ~115 s), P2-concat 8.5, P2-sync-seen-fake 8.5, P2-sync-only 7.0, P2-artifact-only 5.6 |
| Test | ~2–10 s of head inference per model on 820 samples (excluding cache loading) |
| Trainable parameters | B-FATE 2.59 M; B-AVH 0.34 M; AVH-realrecon 0.36 M; P2 0.48 M; P2-concat 0.49 M; P2-sync-seen-fake 0.73 M; P2-sync-only 0.12 M; P2-artifact-only 0.46 M; peak GPU memory during training ≤ 70 MB |
| Generating 910 pairs | **Not formally recorded** (session logs suggest ~50 s/pair effective on 2 GPUs → ~12–13 h estimated, over 2 sessions). Recorded in `generation.json → sessions` since 06/10 |
| Worker installation | MuseTalk/AV-HuBERT ~10–20 min per session (environment ~14 GB in `/kaggle/temp`) |

**Projection for one part the size of part1 (Kaggle GPU hours, 2×T4):**

| Step | Hours | Sessions |
|---|---|---|
| Generation (Wav2Lip, MuseTalk in separate sessions) | ~12–13 | 2 |
| Prepare (FATE / AV-HuBERT + DINOv2) | ~11 + ~1 installation | 2 |
| Train + test 8 architectures × 3 seeds | ~3–3.5 | 1 |
| **Total** | **~27–29 h ≈ one week of quota (30 h)** | 5 |

- Extraction and generation costs **scale linearly with the number of samples**. A part2 twice the size of part1 needs ~2 weeks of quota, or less if only the needed encoders are extracted (dropping FATE if B-FATE is no longer a main baseline).
- Training is cheap (≤ 3.5 h), so new architectures (v002) cost only a few extra hours **as long as the cache is reused**. A new encoder or preprocessing step pays the extraction cost again.
- Training on several parts together (`DATA_PARTS` with several parts) does not re-extract old parts, only new ones; but it needs a new `RUN_NAME` and training from scratch.

## 8. Discussion

### 8.1. By research question

- **RQ1 — Performance.**
  - B-AVH is best overall (AUC 0.837 ± 0.025; seed ensemble 0.849) and the most robust when the generator changes.
  - P2 is strongest on the seen generator (Wav2Lip 0.979), has **the best temporal localisation** (AP@0.5 0.420; short recall 0.44; boundary MAE ~0.2 s) and a low false-alarm rate (0.19).
  - B-FATE is at chance on MuseTalk (0.494) and flags 61% of reals/shams: thresholds chosen on validation do not transfer to new speakers.
- **RQ2 — Does the sync branch contribute?** **No positive contribution could be measured.**
  - P2 − P2-artifact-only = −0.020 overall, −0.044 on MuseTalk, ≈ 0 on Wav2Lip, so P2's performance **comes mostly from the artifact branch**.
  - The gate even reduces generalisation: P2-concat beats P2 by +0.037 on MuseTalk.
  - P2-sync-only is near chance for the AI label (0.55) because it also fires on shams (FAR 0.88). The sync branch does its job of "detecting audio mismatch", but audio mismatch does not mean AI.
- **RQ3 — Artifacts or consistency?**
  - P2 and its variants detect **source ≈ donor** fakes (W2L 0.98/0.98), so the main evidence is **artifacts**.
  - B-AVH shows **donor > source** (W2L 0.938 vs 0.844; MT 0.825 vs 0.757), so it also uses audio–mouth mismatch signals.
  - Shams are not treated as AI by P2 (sham FAR = real FAR = 0.191): P2 does not mistake ordinary audio mismatch for a deepfake.
  - The sync branch scores below 0.5 on W2L source (0.436): fakes painted to the original audio are **better synced than reals**, because Wav2Lip is optimised against a lip-sync expert.
- **RQ4 — Unseen generator.**
  - The artifact branch drops sharply (0.98 → 0.64), while the sync branch is more robust (MT donor 0.80, source 0.74). This **matches the hypothesis "consistency is more robust across generators"** at the branch level.
  - However, the fusion relies on the artifact branch, so fused P2 is only 0.55–0.56 on MuseTalk; B-AVH (no separate branches) keeps 0.79.
- **RQ5 — Freezing the sync branch.** It does not help: P2-sync-seen-fake beats P2 (P2 is −0.028 overall, −0.063 on MuseTalk) and has the lowest false-alarm rate (0.114).

**Part1 conclusion:** the hypothesis "P2 generalises better than B-AVH" is **not supported** (P2 − B-AVH = −0.083 [−0.113; −0.055] overall, −0.255 on MuseTalk). The sync branch carries a generator-robust signal, but (i) the A→V reconstruction foundation did not learn (§7.3) and (ii) the fusion does not use it. P2's real contributions are **temporal localisation** and a **low false-alarm rate** on the seen generator.

### 8.2. Limitations (to be stated with every number)

1. **2 test speakers, 3 validation, 5 train**: not enough to claim generalisation to new speakers; per-source-clip CIs are optimistic (the code's per-group CI needs ≥ 5 groups — there are 2).
2. One held-out generator; training on Wav2Lip only.
3. Stage A failed, so the sync branch cannot be interpreted as "consistency".
4. Speaker imbalance (15–158 clips); speaker_09 makes up 39% of train.
5. The real/fake processing difference has not been checked (Wav2Lip paints a 96×96 face and upscales it), so the artifact branch may learn processing traces.
6. Scores are not calibrated; branches have no thresholds of their own.

### 8.3. Notes on the pipeline

- **Runs end to end on Kaggle** with real data: generating 910 pairs, extracting 3 encoders, training and testing 24 detectors. Resume and signature locks (plan, generation, cache, test-lock) all worked, and caught code changes between sessions.
- **The bottleneck is GPU time, not training**: generation + extraction take ~90% of the cost (§7.9). The weekly quota sets the pace for adding parts.
- **Face rules differ between cutting and generation** (YuNet on ≥ 90% of frames vs S3FD on every frame), so 4/910 pairs were dropped. The rate is low but may bias fakes towards "easy" faces.
- **Kaggle limits cause friction**: Output ≤ 20 GB, datasets made from Output limited to 500 files, the CLI only fetches the latest version, the new CLI needs Python ≥ 3.11. Mitigations: standardised paths, Output keeps only result folders, `tools/kaggle_pull.py`, cache uploaded as local ZIPs.
- **Data is the biggest weakness**: 10 speakers, 1.2 h of real video; every architecture comparison is limited by the number of independent test groups.
- The 2×2 design + per-branch reporting **already answered the artifact-vs-consistency question** on part1 — an evaluation contribution worth keeping.

### 8.4. Summary paragraph

v1 uses 575 real clips from 10 people / 13 videos (1.2 h) and generates 2954 samples with a 2×2 design (Wav2Lip for training, MuseTalk held out).
- On the test set (2 people), B-AVH reaches ROC-AUC 0.837 ± 0.025 and keeps 0.79 on the unseen generator.
- P2 reaches 0.766 ± 0.015 (0.979 on Wav2Lip, 0.553 on MuseTalk), with localisation AP@0.5 of 0.420 and a false-alarm rate of 0.19.
- Per-branch analysis shows that P2's performance comes mainly from mouth-region artifacts. The sync branch is more robust across generators, but the fusion does not exploit it and the A→V reconstruction foundation did not learn.
- Limitations: only 2 test speakers, one held-out generator, and the real/fake processing difference not yet checked.

## 9. Open issues (condensed, 06/10/2026)

**Already handled in v001:**
- duplicate video detection (keep the first, `skipped_videos.csv`);
- YouTube AI-dubbed audio tracks (check `audio_is_original`);
- HTTPS-only downloads; normalisation to 25 fps / ≤ 1080;
- face loss in generators (drop the pair, record the reason, stop after 10 consecutive drops);
- MuseTalk missing one audio frame (10 ms padding);
- split moved to export; one generator per session; review defaults to keep;
- workers installed outside Output; compact Output; test right after training; per-part comparison.

| Area | Open issue | Level | Plan |
|---|---|---|---|
| Speakers | `speaker_id` assigned manually per video; not yet checked that the face in each clip is the right person, nor dubbed voices or off-screen voices | High | v002: ArcFace face clustering + ECAPA voice; active-speaker detection as a hint only |
| Scale | 10 speakers, 2 test speakers; imbalance of 15–158 clips/speaker | **High** | Part2+: target 12–20 people, 2–3 videos each; consider a clips-per-speaker cap |
| Sources | Mixed languages; re-uploads with different IDs not caught; videos removed after collection; very different source compression (0.8–3.3 Mbps) | Medium | Per-clip language ID; near-duplicate detection; keep raw + hashes; R3 |
| Cutting | VAD fragments speech (0.3 s silence merging) → ~45% of speech becomes clips; face rules looser than the generators'; occluded mouths, overlaid text, background music, beauty filters; existing real A/V offsets not measured | Medium–**High** | Merge silences of 0.5–1.0 s in `data.yaml`; S3FD on every frame at cutting; measure real A/V offsets **for reporting only** |
| Review | A single reviewer; the eye struggles below 100 ms offsets | Medium | Cross-review a random sample |
| Generation | Shams only splice audio at the same position; fake A/V offsets not measured (`null`); real/fake processing difference | **High** | More sham types; controls: real re-encoded / passed through a generator in reconstruction mode |
| Features | AV-HuBERT pretrained on English; mouths of ~30–60 px upscaled to 224 for DINOv2 | Medium | R1; see §7.3 |
| Models | Stage A no better than constant; fusion ignores the sync branch; scores not calibrated, branches without thresholds | **High** | v002 (§10) |
| Cost | Generation/extraction ~25 h GPU per part1-sized part | Medium | Separate sessions; extract only needed encoders; log `sessions` for projections |
| Infrastructure | yt-dlp dropping Python 3.10; new code not committed; demo not run with real checkpoints | Low | Move the data env to ≥ 3.11; commit |
| Ethics | Copyright/consent/privacy of people in YouTube videos | Medium | Publish only IDs/URLs + reproduction scripts; public demos use consenting people |

## 10. Next steps

### 10.1. Finishing v001 (no research change)

1. Commit the code; push so the Kaggle notebooks clone the new version.
2. Run the local demo with real B-AVH/P2 checkpoints.
3. Draw report figures (pipeline/P2 diagrams, 2×2 statistics, real/sham/source/donor timelines with both branches, FP/FN examples).

### 10.2. v002 — research

| # | Direction | Reason (from v001) |
|---|---|---|
| 1 | **Fix the sync branch**: replace A→V reconstruction with a direct synchronisation signal (e.g. similarity/offset between AV-HuBERT audio and visual embeddings over time, or a SyncNet-style head [19] learning real vs shifted) | Stage A no better than constant; a measurable consistency foundation is needed |
| 2 | **Fusion that the artifact branch cannot dominate**: concat/late fusion, artifact-branch dropout or gate constraints; report per-branch scores with their own thresholds | The gate is worse than concat; the sync branch is more robust but ignored |
| 3 | **Data**: part2+ (12–20 people), speaker balance; automatic speaker assignment | 2 test speakers, insufficient CIs |
| 4 | **Several generators at training time** (e.g. a part generated entirely with MuseTalk, train Wav2Lip + MuseTalk, a third held-out generator) — new protocol, separate `RUN_NAME` | The artifact branch memorises Wav2Lip |
| 5 | **Check the real/fake processing difference** (R3 question 1) before reading the artifact branch | Risk that artifact = processing trace |
| 6 | Score calibration (temperature scaling [22] / isotonic regression on validation); per-branch thresholds | High false alarms for B-FATE/AVH-realrecon |
| 7 | R1–R3 (resolution, fps, compression) — see below | No evidence yet |

**R1. Download and processing resolution.**
- Current setting: download ≤ 1080 short side, reject < 720; models process a 640 long side.
- To measure: actual mouth-box size; the artifact branch at `max_side` 640 vs 960/1280 (AV-HuBERT and DINOv2 must share `max_side`).

**R2. Source fps and normalisation to 25 fps.**
- Current setting: reject < 25 fps, normalise to 25 fps CFR.
- To measure:
  - sync-branch scores on reals by source fps group (25/50 vs 30/60);
  - 24 fps sources tried as a side set;
  - the distribution of existing audio–visual offsets in reals (reporting only).

**R3. Compression.**
1. Real/fake compression difference: controls with reals re-encoded once more, or passed through a generator in reconstruction mode.
2. Scores by source bitrate/codec group.
3. Stress tests with extra compression at several CRF levels.

Data available: `video_metadata.csv`, `sources.jsonl`, and `source_codec` / `source_bitrate_kbps` per clip.

### 10.3. Closing v001 (14/10/2026)

| Item | Status | Note |
|---|---|---|
| §10.1.1 Commit / push | ✔ | The 06/10 overhaul and v002 were committed (`b6ff3a7`, `f969cb7` and later commits) |
| §10.1.2 Local demo with real checkpoints | ✘ carried over | v001 checkpoints exist only in the archive; demo with v002 checkpoints once a model is chosen |
| §10.1.3 Report figures | ✘ carried over | The v001 slides have diagrams; result figures go in the final report |
| §10.2.1 Fix the sync branch | ✔ in v002 | An offset-classification head replaced A→V reconstruction. The "mismatch?" output reaches sham 0.943, MuseTalk donor 0.826 (v002 §15.7) |
| §10.2.2 Fusion not dominated by the artifact branch | ✔ in v002, negative result | Late fusion with non-negative weights; the "AI-painted?" output is still ≈ artifact because sync carries no "AI" information (v002 §15.7) |
| §10.2.3 Part2 data | Open | Still the largest limitation (v002 test has only 3 speakers) |
| §10.2.4 Several generators at training time | Replaced by MechFake | Training without any generator (v002) |
| §10.2.5 Real/fake processing difference | ✔ in v002 (E6) | **Risk confirmed:** the artifact branch flags benign processing (v002 §15.6) |
| §10.2.6 Calibration | Open | v002 §19 items 3 and 11 |
| §10.2.7 R1–R3 | Open | R1 remains for v002 (mouth boxes 45–85 px); R3 relates to E6b |

**v001's conclusions seen from v002:**
- "The artifact branch memorises its generator" (§8.1 RQ4) was sharpened by v002: it learns **processing traces**. Models trained on Wav2Lip flag 89–95% of real clips whose mouth was merely pasted back.
- "B-AVH is the most robust" still holds: B-AVH is the only model that passes v002's processing diagnostics.

---

# PART II — TECHNICAL DETAILS

## 11. Real-data pipeline (`data_pipeline/`)

Runs locally in conda `vn-av-df` (Python 3.10).
- The part is selected with `PART` in `data_settings.py`; part names are `vn-av-df-data-part<N>`.
- Cutting can run on Kaggle (`cut.ipynb`, same code/config, signed).

### 11.1. Folder layout per part

| Step | Script | Folder |
|---|---|---|
| 00 setup | `steps/00_setup.py` | Downloads YuNet, Silero VAD |
| 01 collect | `steps/01_collect.py` | `data/sources/<dataset>/<part>/` — `videos.csv`, `selected_videos.csv`, `video_metadata.csv`, `skipped_videos.csv` |
| 02 download | `steps/02_download.py` | `data/raw/<dataset>/<part>/` — `videos/*.mp4`, `sources.jsonl`, `logs/` |
| 03 cut | `steps/03_cut.py` | `data/candidates/<dataset>/<part>/` — `clips/*.mp4`, `review.csv`, `logs/` |
| 04 review | `steps/04_review.py` | Review web app at `127.0.0.1:8001`, writes `review.csv` |
| 05 export | `steps/05_export.py` | `exports/<dataset>/<part>/` — kept clips, `manifest.jsonl` (**with `split`**), `dataset_info.json` (**with `split`**), `split-lock.json`, `quality_report.json` |

Matching Kaggle datasets: `vn-av-df-raw-partN` (raw) and `vn-av-df-clean-partN` (export). To extend a part, add URLs and rerun 01 → 05; each step processes only the new part.

### 11.2. 01 collect (`collect.py`)

1. **Expand and deduplicate.**
   - `videos.csv` holds only `url,speaker_id`; a playlist gives every video the row's `speaker_id`.
   - For duplicates the first occurrence is kept; videos already in another part are dropped.
   - Every unselected video is recorded in `skipped_videos.csv` (`type` = `same_part`/`other_part`/`rejected`/`error`, `reason`, `speaker_conflict`).
2. **YouTube metadata** once per video, applying step 02's format-selection rules offline. Written to `video_metadata.csv`: identifiers, original vs to-be-downloaded format, audio tracks (`audio_track_count`/`audio_is_original`), status.
3. **Quality check** (metadata only). Rejects:
   - livestreams and non-public videos;
   - duration < 5 s or > 12 h;
   - **original fps < 25**;
   - no HTTPS mp4 ≥ 25 fps + m4a;
   - **short side < 720**.
   For 4K sources, the available 1080 version is used.

`selected_videos.csv` has fixed columns `url, video_id, speaker_id, title, channel, playlist_url` and is append-only. Changing the `speaker_id` of a selected video raises an error.

### 11.3. 02 download

- yt-dlp + Node.js.
  - Format: `bestvideo[ext=mp4][protocol=https][fps>=24.5]+bestaudio[ext=m4a][protocol=https]`, `format_sort = ["res:1080"]`.
  - **HTTPS only** (no HLS).
- Files: `videos/<video_id>.mp4`.
  - Downloads resume from `.part`; failed merges are deleted.
  - `sources.jsonl` records `source_id = yt_<video_id>`, the SHA-256 and the file's real properties.
- Quality is re-checked on the file. **No re-encoding** at this step: raw is the faithful archive for R1–R3, and cutting is the only encode.

### 11.4. 03 cut (`curation.py`, `configs/data.yaml`)

1. **Timeline:** one shared time origin (container `start_time`) for video and audio.
2. **Speech:** Silero VAD [13] v6, mono 16 kHz, threshold 0.5; silences ≤ 0.3 s merged; segments < 0.25 s dropped; 0.15 s padding.
3. **Visual:** scanned at 4 fps with long side ≤ 640.
   - Scene cuts: 8×8×8 histogram correlation < 0.55.
   - Faces: YuNet [12], exactly one face ≥ 64 px.
4. **Windows:** 5–8 s inside each speech ∩ scene segment, 1 s overlap, on the 25 fps grid. Segments < 5 s are dropped (matching the parent ≥ 5 s threshold).
5. **Face filter:** ≥ 90% of scanned frames pass.
6. **Encode:** x264 CRF 18, AAC 160 kbps 48 kHz, **25 fps CFR**, short side ≤ 1080. Records `source_fps/width/height/codec/bitrate_kbps`.
7. **Output and reruns:**
   - Clips are written with `decision = uncertain`.
   - `logs/sources.json` records each video (kept clips, rejected windows + reasons).
   - Reruns cut only videos not yet cut; changed rules/code/models raise an error.

### 11.5. 04 review and 05 export

- **Review page** (shared with generated-sample review, `vn_av_data/serving/review_page.py`):
  - searchable, filterable list (decision, speaker, source, quality) and a progress bar;
  - shortcuts `K`/`R`/`U`, `←/→`, `N` (next uncertain sample), `Space`, `L` (loop), speed control;
  - saves to CSV immediately.
- **Export:**
  - only `keep` clips; the whole part is rebuilt in `.partial`;
  - **assigns the split** (§12), then writes `split` into the manifest, `split-lock.json` and `dataset_info.split` (`data_part`, `split_ratios`, `seed`, `covered_parts`, `groups`, `counts`);
  - an unchanged manifest keeps the old copy; a changed one replaces it, and the part's generation must be redone.
- Data contract `vn_av_data.contract.validate_bundle`: if `split` is present, every clip must have a valid one; no `variant`.

## 12. Splitting (`vn_av_data/data/split.py`, run at 05 export)

1. **Connected groups (union-find)** over:
   - `source_id`, media SHA-256, `source_sha256`, `duplicate_group_id`;
   - `parent_ids`, `speaker_id(s)`, `global_speaker_ids`, `group_id`.

   Sharing any node → same group → same split. `vn_av_df.data.groups` reuses the same function to check leakage when loading data and for bootstrapping.
2. Fewer than 3 groups (without history) → error.
3. Groups are shuffled by `SPLIT_SEED`, largest first. Each group is assigned to the split that minimises the sum of squared deviations from the targets; every split gets at least one group.
4. Ratios `SPLIT_RATIOS` (default 70/15/15); each part is split independently (`SPLIT_HISTORY = []`). Inheritance is enabled with the `split-lock.json` of every earlier part.
5. Generation only **reads** the clean part's split (the plan errors if the part has none) and copies `split-lock.json` into the generated part.

Hashes only catch byte-identical duplicates; reposts/near-duplicates need a `duplicate_group_id` assigned during review.

## 13. Fake generation and labels (`generation.py`, `generate.ipynb`)

### 13.1. Plan (`02_plan`)

- **Parents:** clean clips **≥ 5 s**; `clips_per_split = 0` takes all (2 = pilot).
- **Donor:** another clip, **same split, same `speaker_id`**, at least as long as the parent, from a different source or non-overlapping in time. Chosen by seed.
- **Audio modes by split:** train `[donor]`; val/test `[donor, source]`. Parents without a donor are dropped.
- **Generators by split:** train/val `[wav2lip_gan]`; test `[wav2lip_gan, musetalk_1_5]`.
- **Per parent:** 1 real + 1 sham (if a donor exists) + for each generator × mode a full and a partial sample. Real/sham come with the first job (Wav2Lip).
- The plan locks parents/donors/split/config and stores the clean part's `split`.

### 13.2. Generating each pair (`03_generate`)

1. **Decode the parent:** 25 fps, 48 kHz PCM on the same PTS origin, long side ≤ 640.
2. **Generator workers** live for the whole run (one per GPU):
   - **Wav2Lip GAN** [6] (`Wav2Lip-SD-GAN.pt`, S3FD [14], 16 kHz audio as upstream), lossless intermediate.
   - **MuseTalk 1.5** [7]: VAE ft-mse [8] 256 crop, Whisper-tiny [9], DWPose [10] / BiSeNet face parsing [11], one-step UNet 1.5. Separate environment; 10 ms audio padding (upstream rounding bug).
3. **Audio reassignment:** source → original PCM; donor → donor PCM trimmed to the exact length. The generator's PCM is never used.
4. **Partial:** 0.4/0.8/1.6/2.4 s at a random position; exactly the frames and PCM are replaced (1920 samples/frame).
5. **Sham:** original visual; PCM in the same partial interval replaced with the sham donor's PCM at the same position. AI = 0.
6. **Identical final encode:** H.264 + AAC, CRF 18 (`common-pcm48k-h264-aac-crf18`).
7. **Generators chosen per session** (`generation_generators`, notebook `GENERATORS`):
   - other generators' pairs are left for later sessions (`PREVIOUS_OUTPUT`), and only the needed workers are installed;
   - provenance is checked per generator (changing a checkpoint between sessions raises an error);
   - each session writes `generation.json → sessions` (time, pair count, seconds per pair by generator).
8. Resume via `records/<key>.json`. When `TIME_BUDGET_HOURS` runs out → `status: partial` + `remaining_by_generator`.
9. **Failures:**
   - a generator failure on a single pair (no face → `face`, other → `generator_error`) drops the whole pair and is recorded in `skipped_pairs.json`;
   - a dead worker, a timeout, a pipeline error, or 10 consecutive dropped pairs → stop.

### 13.3. Main label fields (schema `vn-av-df-v1`)

| Field | Real | Sham | Fake full | Fake partial |
|---|---|---|---|---|
| `label` | 0 | 0 | 1 | 1 |
| `fake_intervals` | `[]` | `[]` | `[[0, T]]` | `[[a, b]]` |
| `av_mismatch_intervals` | `[]` | `[[a, b]]` | `null` | `null` |
| `audio_mode` | null | null | `source`/`donor` | `source`/`donor` |
| `audio_edit_kind` | none | splice | none / donor_replace | none / splice |
| `control_type` | null | `conventional_audio_splice` | null | null |
| `sync_status` | reviewed_match | unknown | unknown | unknown |

Also recorded: `source_id`, `source_clip_id`, `speaker_id(s)`, `parent_ids`, `audio_source_id`, `generator`, `generator_version`, `generator_checkpoint_sha256`, `variant`, `group_id`, `sha256`. 2×2 cells (`dataset.condition`): `real`, `sham`, `<generator>/<audio_mode>`.

### 13.4. Review, finalise, package

- **`04_review`** (shared UI, §11.5):
  - **every sample defaults to keep**; only broken samples are marked reject/uncertain;
  - the original real plays side by side, in sync;
  - the timeline marks fake intervals (red) and sham splices (orange);
  - filters by split/variant/generator/audio_mode/speaker.
- **`05_finalize`:** only keep samples go into `manifest.jsonl`; every split has reals and fakes; immutable.
- **`06_export`:** ZIP `vn-av-df-data-partN.zip`. The part folder is kept inside, because the folder name enters the selection hash.

### 13.5. Checks at load time (`dataset.validate_rows`)

- unique `sample_id`; valid split;
- labels consistent with `fake_intervals`; valid intervals;
- reals have no A/V mismatch;
- the same hash never crosses two splits; **connected groups never cross two splits**;
- when merging parts, the same media in two parts → error.

Notebooks take full part paths (`part_selection`); the part name (last folder) determines the selection hash.

## 14. Feature extraction (`prepare.ipynb`, `training/01_prepare_features`)

Each backbone is extracted **once** for all models that use it.
- Caches are bound to the video hash + signature (asset, code, options).
- Encoders are chosen with `ENCODERS` (`prepare_encoders`).
- Old caches attach via `CACHE_INPUTS`: the encoder being extracted is copied to continue; other encoders are symlinked read-only.
- Extraction runs in parallel on all GPUs (`prepare_devices`). AV-HuBERT uses 2 workers/GPU, and samples of the same parent go to one worker to reuse landmarks.

### 14.1. FATE (B-FATE) [1, 2]

- Decode ≤ 60 s, side ≤ 640; YuNet single face.
- 2 s windows with a 0.2 s step; PE-AV Small + FATE adapter frozen, fp16.
- Video preprocessed once per clip, `batch_windows` 8.
- Descriptors: 8 temporal bins, separately for A and V; minimum coverage 0.75.

### 14.2. AV-HuBERT (B-AVH, P2 sync branch, `avh_realrecon`) [3, 4]

Base LRS3 without fine-tuning (`base_lrs3_iter4.pt`); a separate Python 3.10/fairseq worker running continuously (`--serve`).
1. Decode ≤ 60 s, side ≤ 640, 25 fps, PCM 16 kHz.
2. Dlib [15, 16] 68 landmarks per frame (LRU keyed by frame hash), exactly one face; no interpolation across face-loss gaps.
3. 13-frame smoothing within valid runs; similarity transform to the mean face [17] at 256×256.
4. Lip ROI cropped 96 → 88, greyscale. **Mouth box for DINOv2:** centre of the lip bbox, side = max(width, height) × 1.6 (`mouth_boxes`).
5. Audio: 26-band log filterbank, stacked by 4 → 104 dims per 40 ms.
6. Chunks of 200 frames (50 overlap), A-only and V-only runs separately → `audio[T,768]`, `visual[T,768]` at 25 Hz; `output_stride = 5`.

### 14.3. DINOv2 mouth crops (artifact branch) [5]

- `facebook/dinov2-small`, fp16, batch 64.
- Square crops from `mouth_boxes` → 224×224 RGB.
- Features [CLS ‖ mean patch] = 768 dims at 25 Hz; `box_sha256` binds them to the AV-HuBERT cache.

## 15. Architectures

**Notation:** T frames at 25 Hz; C = ⌈T/5⌉ cells of 0.2 s.
- Temporal blocks run **separately on each contiguous observed run**.
- A cell is valid when all its frames are valid; invalid cells get logit 0 and are excluded from the loss.
- `TemporalBlock(d)` = `x + Dropout(GELU(LayerNorm(Conv1d(k=3, dilation=d)(x))))`.

### 15.1. B-FATE — `fate_gru`

`A[C,Da] ‖ V[C,Dv] → Linear(→128) → LayerNorm → GELU → 2-layer BiGRU (128/direction, dropout 0.1) → Linear(256→1)`.

### 15.2. B-AVH — `avh_tcn`

`AV-HuBERT A,V → average over 5 frames → A‖V (1536) → Linear(→128) → LayerNorm → GELU → 3 residual blocks [Conv1d k3 d=1/2/4 → GELU → Dropout] → Linear(128→1)`. No reconstruction, no separate sync branch.

### 15.3. P2 — `p2_syncartifact`

**Sync branch:**
- normalise a, v (μ, σ fitted on real train);
- `R: a → Linear(768→128) → GELU → TemporalBlock×3 → Linear(128→768) = v̂`;
- `e = [|v − v̂| ‖ MSE ‖ 1 − cos]` (770) → average over 5 frames → `Linear(770→128) → LN → GELU → TemporalBlock×3 = h_sync → Linear = logit_sync`.

**Artifact branch:** `[x ‖ Δx]` (1536, Δ between consecutive frames) → `Linear(→128) → LN → GELU → TemporalBlock×3` (25 Hz) → average over 5 frames = `h_art → Linear = logit_art`.

**Fusion:**
- gate `w = softmax(MLP([h_sync ‖ h_art]))`, `f = w₀·h_sync + w₁·h_art`;
- `[f ‖ logit_sync ‖ logit_art] → Linear(130→128) → LN → GELU → TemporalBlock×2 → Linear = logit_AI`;
- both branch logits are also output separately (`outputs`) for the timelines and `branches`.

| Stage | Learns | Data | Objective | After the stage |
|---|---|---|---|---|
| **A** | μ/σ and R | Real train `reviewed_match` | SmoothL1(R(a), v); the constant-0 predictor loss is recorded as reference | Freeze R, μ, σ |
| **S** | Projection + TCN + sync head | Real + sham (+ real shifted ±3–15 frames, p = 0.5); **no fakes** | Per-cell BCE on the A/V mismatch label | Freeze the sync branch |
| **C** | Artifact branch + gate + fusion | All of train | BCE on the AI label + 0.5 × artifact BCE | — |

**Variants** (same `SyncArtifactDetector` class; unused branches have gradients disabled):
- `p2_concat` (gate → Linear 256→128);
- `p2_sync_only`, `p2_artifact_only` (no stages A/S);
- `p2_sync_seen_fake` (sync branch not frozen in stage C).

### 15.4. `avh_realrecon`

- Stage A as in P2.
- Visual (pooled over 5) and residual (770) → each through Linear→128/LN/GELU; a gate mixes them; 3 TemporalBlocks → Linear→1.
- No stage S, no DINOv2. (The old `concat` class is kept frozen only to load trained checkpoints.)

## 16. Training (`experiment.train_one`)

| Item | Rule |
|---|---|
| Unit | One clip per step |
| Sampling | `balance_parents`: each epoch takes one random variant per parent, then shuffles |
| Cell target | Fraction of the cell's time altered by AI (0–1) |
| Loss | Per-cell BCE + video BCE (video score = mean of the top 10% valid cells) |
| Optimiser | AdamW [21] lr 3e-4, wd 1e-4, clip 5; hidden 128, dropout 0.1 |
| Budget | 20 epochs, patience 5; `best.pt` by validation loss |
| Resume | `run_id` = hash(data, cache, configuration, model, seed, code); a mismatch → error |
| Held-out | `protocol_rows` removes MuseTalk from train/val |
| Several parts | Manifests merged in memory; changing the part set → new `RUN_NAME` |

`train.ipynb` downloads no encoders/workers. It attaches caches by symlink and checks that `extraction.json` covers the needed encoders on the right data hash.

## 17. Evaluation (`experiment.evaluate`, `metrics.py`, `compare.ipynb`)

1. **Thresholds** (validation): video and temporal, separately, at max F1 (ties → lower FPR → higher threshold).
2. **Video:** ROC-AUC, AP (`pr_auc`), F1, precision, recall, FPR, confusion matrix.
3. **Intervals:** consecutive cells ≥ the temporal threshold → one interval, scored by the cell mean.
4. **Temporal AP** at tIoU 0.3/0.5/0.75/0.95 (one-to-one matching), with `matched`, mean tIoU, boundary MAE, short recall (≤ 1 s).
5. **Alarms on label 0:** share of clips with an interval; intervals per hour.
6. `by_condition` (fake cells + original real; real/sham separately), `branches` (per-branch AUC, real vs cell), `by_generator`, `by_variant`, `by_part`.
7. `uncertainty`: 95% CI of AUC by bootstrap over connected groups (< 5 groups → not reported).
8. **Test-lock:** hashes of the manifest, run index and checkpoints; a change after the first test → error.
9. **`compare.ipynb`:**
   - reads `predictions-test.json` from several runs on the same data;
   - produces **one set of tables per part** (+ merged), with mean ± std, seed ensemble, AUC by generator/cell/branch (sham vs real for branches), temporal AP, paired ΔAUC bootstrapped over source clips, and cost and GPU projections;
   - writes to `reports/compare/`.

## 18. Output artifacts

| Content | Files |
|---|---|
| Run list | `runs/<RUN_NAME>/training.json`, `dataset-selection.json` |
| Each model/seed | `configuration.json`, `history.json`, `best.json`, `best.pt`, `last.pt`, `resources.json`, `wandb.json` |
| Stages A / S | `reconstruction.json`, `sync.json` |
| Validation / test | `validation.json`, `predictions-validation.json`; `test.json`, `predictions-test.json` (since 06/10 also with `source_clip_id`, `speaker_id`, `generator`, `variant`), `evaluation.json`, `test-lock.json` |
| Figures | `report/validation-*`, `test-*`, `reconstruction.png`, `sync.png` |
| Cache | `cache/<dataset>/<encoder>/extraction.json`, `environment/avhubert/` |
| Generation | `generation.json` (plan, provenance, `sessions`), `skipped_pairs.json`, `split-lock.json` |
| Comparison | `reports/compare/<split>_<part>.md` + PNG |

## 19. Demo and inference

`training/04_demo.py` (`127.0.0.1:8000`) + the `demo/` frontend.
- Uses one detector (`DEMO_METHOD`, first seed in `training.json`) or `CHECKPOINT`.
- Uses the same encoder, masking and thresholds as training.
- Outputs `video_score`, intervals, coverage and a timeline; P2 adds `evidence`/`evidence_scores` (no per-branch thresholds yet).
- The "Test results" table reads `evaluation.json` of the selected run.

## 20. Where things run and environments

| Stage | Where | Needs |
|---|---|---|
| Collect / cut / review / export (split) | Local `vn-av-df` (Python 3.10); cutting can use `cut.ipynb` | yt-dlp + Node.js, YuNet, Silero VAD |
| Generation | Kaggle `generate.ipynb`, one generator per session | Wav2Lip; MuseTalk worker (MuseTalk sessions only) |
| Review / finalise / ZIP fakes | Local | Generation Output |
| Feature extraction | Kaggle `prepare.ipynb` | FATE, AV-HuBERT worker, DINOv2 |
| Train + test | Kaggle `train.ipynb` (or local CPU) | Caches only |
| Comparison | Local or Kaggle `compare.ipynb` | numpy, scikit-learn, matplotlib |
| Demo | Local | Encoder + `best.pt`, AV-HuBERT worker for B-AVH/P2, Node.js |

Every notebook:
- takes full input paths;
- keeps only the result folders under `vn-av-df-forensics/` in Output;
- writes W&B logs to `/kaggle/working/wandb`.

Outputs are downloaded with `tools/kaggle_pull.py`.

## 21. Design changes during v001

| Date | Change |
|---|---|
| 29/09/2026 | Initial design: B1 `fate_linear`, B2 `fate_gru`, P1 `avh_realrecon`; fakes only with `audio_mode = source` |
| 30/09/2026 | **Option A** chosen: B-FATE / B-AVH / P2; 2×2 data (donor + sham); DINOv2 mouth crops, stage S, `by_condition`/`branches`, demo with branch timelines. The data pipeline reduced to 5 steps |
| 01/10/2026 | Real part1 download (HTTPS only); minimum clip 5 s; source quality checks (≥ 25 fps, short side ≥ 720, download ≤ 1080, 25 fps CFR clips); YouTube metadata + pre-download checks; R1–R3 |
| 03/10/2026 | `skipped_videos.csv`; `cut.ipynb`; W&B for cut/generate/train |
| 05/10/2026 | **Data:** part1 review done (575 keep, 10 speakers), split 70/15/15. **Generation:** continuous workers, multi-GPU, time budget, dropping pairs with generator errors (face loss, per-pair errors), MuseTalk 10 ms padding, workers outside Output. **Extraction:** FATE/AV-HuBERT sped up (one-time preprocessing, window batching, multi-GPU, continuous workers, landmark cache). **Runs:** part1 generated (2954 samples); B-FATE, B-AVH, P2 trained and tested |
| 06/10/2026 | **Runs:** 5 variants trained and tested; 24 detectors merged into `train_part1_run01`; results and discussion (§7–8). **Notebooks:** `prepare.ipynb` split from `train.ipynb`; `ARCHITECTURES` = `"all"` or a list; full input paths, `PREVIOUS_OUTPUT`, Output keeps only results; `RUN_TEST = True`; `compare.ipynb` per part; `tools/kaggle_pull.py`. **Pipeline:** **split moved to 05 export**; **one generator per session** + session time logs; **generated-sample review defaults to keep**; new shared review UI. **Removed:** old architectures (`fate_linear/tcn/transformer`, `avh_linear/visual/concat`), the old `compare()`/`comparison.json`, multi-model analysis in the demo, external-generator import, old review import, the assign/merge-reviews/index commands; scripts renumbered (`generation/06_export`, `training/04_demo`) |

## 22. References

Only models, tools and ideas **actually used or cited in v001**; pinned versions/commits/checkpoints are in §7.1. `[n]` numbers are cited in the corresponding sections.

### 22.1. Encoders and pretraining data

| # | Reference | Used in |
|---|---|---|
| [1] | A. Vyas, H.-J. Chang, C.-F. Yang, P.-Y. Huang, L. Gao, J. Richter, S. Chen, M. Le, P. Dollár, C. Feichtenhofer, A. Lee, W.-N. Hsu. *Pushing the Frontier of Audiovisual Perception with Large-Scale Multimodal Correspondence Learning.* arXiv:2512.19687, 2025. Model `facebook/pe-av-small`. [arXiv](https://arxiv.org/abs/2512.19687) | PE-AV Small — B-FATE (§14.1) |
| [2] | K. Guan, B. Zhang, X. Wang, Y. Ba, X. Cheng, Y. Chen, R. Song. *FATE: Frame-Level Audio-Visual Temporal Embedding.* arXiv:2608.01310, 2026. Adapter `Guan123/fate`. [arXiv](https://arxiv.org/abs/2608.01310) · [code](https://github.com/guankaisi/FATE) | FATE adapter — B-FATE (§14.1) |
| [3] | B. Shi, W.-N. Hsu, K. Lakhotia, A. Mohamed. *Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction* (AV-HuBERT). ICLR 2022. [arXiv:2201.02184](https://arxiv.org/abs/2201.02184) · [code](https://github.com/facebookresearch/av_hubert) | B-AVH, P2 sync branch, `avh_realrecon` (§14.2) |
| [4] | T. Afouras, J. S. Chung, A. Zisserman. *LRS3-TED: a large-scale dataset for visual speech recognition.* arXiv:1809.00496, 2018. [arXiv](https://arxiv.org/abs/1809.00496) | Pretraining data of `base_lrs3_iter4.pt` |
| [5] | M. Oquab, T. Darcet, T. Moutakanni, et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | `facebook/dinov2-small` (ViT-S/14) — P2 artifact branch (§14.3) |

### 22.2. Generators and components

| # | Reference | Used in |
|---|---|---|
| [6] | K. R. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, C. V. Jawahar. *A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild* (Wav2Lip). ACM Multimedia 2020. [arXiv:2008.10010](https://arxiv.org/abs/2008.10010) · [code](https://github.com/Rudrabha/Wav2Lip) | Wav2Lip GAN, train/val/test generator (§13.2) |
| [7] | Y. Zhang, Z. Zhong, M. Liu, Z. Chen, B. Wu, Y. Zeng, C. Zhan, J. Huang, Y. He, W. Zhou. *MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling.* arXiv:2410.10122, 2025. [arXiv](https://arxiv.org/abs/2410.10122) · [code](https://github.com/TMElyralab/MuseTalk) | MuseTalk 1.5, held-out generator (§13.2) |
| [8] | R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | VAE `stabilityai/sd-vae-ft-mse` in MuseTalk |
| [9] | A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, I. Sutskever. *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). ICML 2023. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) | `openai/whisper-tiny` — MuseTalk audio features |
| [10] | Z. Yang, A. Zeng, C. Yuan, Y. Li. *Effective Whole-body Pose Estimation with Two-stages Distillation* (DWPose). ICCV 2023 Workshops. [arXiv:2307.15880](https://arxiv.org/abs/2307.15880) | Face landmarks in MuseTalk |
| [11] | C. Yu, J. Wang, C. Peng, C. Gao, G. Yu, N. Sang. *BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation.* ECCV 2018. [arXiv:1808.00897](https://arxiv.org/abs/1808.00897) | Face parsing (ResNet-18 backbone) in MuseTalk |

### 22.3. Face detection, speech, landmarks

| # | Reference | Used in |
|---|---|---|
| [12] | W. Wu, H. Peng, S. Yu. *YuNet: A Tiny Millisecond-level Face Detector.* Machine Intelligence Research 20(5):656–665, 2023. [DOI](https://doi.org/10.1007/s11633-023-1423-y) · [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Face filtering at cutting (§11.4), faces for FATE (§14.1) |
| [13] | Silero Team. *Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier.* GitHub, 2024 (v6.0 used). [code](https://github.com/snakers4/silero-vad) | Speech segmentation at cutting (§11.4) |
| [14] | S. Zhang, X. Zhu, Z. Lei, H. Shi, X. Wang, S. Z. Li. *S³FD: Single Shot Scale-invariant Face Detector.* ICCV 2017. [arXiv:1708.05237](https://arxiv.org/abs/1708.05237) | Face detection in Wav2Lip (§13.2) |
| [15] | D. E. King. *Dlib-ml: A Machine Learning Toolkit.* JMLR 10:1755–1758, 2009. [dlib.net](https://dlib.net) | Detector + `shape_predictor_68_face_landmarks` (§14.2) |
| [16] | V. Kazemi, J. Sullivan. *One Millisecond Face Alignment with an Ensemble of Regression Trees.* CVPR 2014. | Algorithm of dlib's 68-landmark predictor |
| [17] | B. Martinez, P. Ma, S. Petridis, M. Pantic. *Lipreading using Temporal Convolutional Networks.* ICASSP 2020. [arXiv:2001.08702](https://arxiv.org/abs/2001.08702) · [code](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks) | `20words_mean_face.npy` — mean-face alignment for the lip ROI (§14.2) |

### 22.4. Method and training ideas

| # | Reference | Used in |
|---|---|---|
| [18] | C. Koutlis, S. Papadopoulos. *AuViRe: Audio-visual Speech Representation Reconstruction for Deepfake Temporal Localization.* WACV 2026. [arXiv:2511.18993](https://arxiv.org/abs/2511.18993) · [code](https://github.com/mever-team/auvire) | Inspiration for the A→V reconstruction residual (stage A, P2 and `avh_realrecon`, §15.3–15.4); **not a reproduction** |
| [19] | J. S. Chung, A. Zisserman. *Out of Time: Automated Lip Sync in the Wild* (SyncNet). ACCV 2016 Workshops. | Direct sync-head direction for v002 (§10.2); why reals are not filtered with SyncNet (§9) |
| [20] | S. Bai, J. Z. Kolter, V. Koltun. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* (TCN). arXiv:1803.01271, 2018. [arXiv](https://arxiv.org/abs/1803.01271) | Residual dilated convolution blocks (`avh_tcn`, `TemporalBlock`, §15) |
| [21] | I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization* (AdamW). ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) | Optimiser (§16) |
| [22] | C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) | Temperature scaling — score calibration in v002 (§10.2) |

### 22.5. Software

| Tool | Used in |
|---|---|
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Metadata queries, YouTube downloads (§11.2–11.3) |
| [FFmpeg](https://ffmpeg.org) (libx264, AAC) | Decoding/encoding clips and generated samples (§11.4, §13.2) |
| [OpenCV](https://opencv.org) | Running YuNet, frame processing |
| [PyTorch](https://pytorch.org), [Hugging Face Transformers / Hub](https://huggingface.co) | Encoders, detectors; downloading revision-pinned checkpoints |
| [fairseq](https://github.com/facebookresearch/fairseq) | AV-HuBERT worker environment |
| [scikit-learn](https://scikit-learn.org) | ROC-AUC, AP (§17) |
| [Weights & Biases](https://wandb.ai) | Logging for cut/generate/prepare/train |
| [Kaggle](https://www.kaggle.com) | GPU runs on 2×T4 (§20) |
