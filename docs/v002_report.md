# v002 — MechFake: detecting Vietnamese lip-sync deepfakes without generator fakes at training time

Updated **14 October 2026**. Status: **part1 experiments complete** (20 configurations × 3 seeds, trained and tested once).

How this report is organised:
- §0–§14 are the original proposal and its implementation, kept unchanged for reference.
- **Results are in §15, discussion in §16, limitations in §17, the direction decision in §18, and follow-up research in §19.**

It builds on the part1 results of v001; the v001 report is `docs/v001_report.md` on branch `archive/v001-part1-results`.

Conventions:
- Numbers marked "v001" are **measured** on the v001 part1 test set.
- GPU costs, parameter counts, thresholds and expectations are **estimates or proposals**.
- "Not found" means not found as of 12 October 2026; it does not claim nobody has done it.

---

## 0. Summary

Current lip-sync detectors learn from the fakes of a few generators and therefore memorise those generators. v001 measured this: the artifact branch of P2 reached 0.98 on Wav2Lip but only 0.55–0.64 on the unseen MuseTalk.

v002 proposes **MechFake**: fake training data created **only from real videos**. Each type imitates one **mechanism** that lip-sync generators use to produce a mouth:
- painting at low resolution and upscaling;
- decoding from a VAE latent;
- warping a reference image;
- splicing at the wrong time;
- pasting.

The detector **TriCue** has 3 branches (sync, lead–lag, painting trace) and 2 outputs ("audio–visual mismatch?" and "AI-painted?"). It is trained **without fakes from any generator**; Wav2Lip and MuseTalk appear only at test time.

Expected contributions:
1. Pseudo-fakes of the mouth region **per generation mechanism** (downscale–upscale, VAE, warp), with a **mechanism → generator map** from leave-one-mechanism-out experiments.
2. A Vietnamese lip-sync detector trained without any generator, **with localisation of fake segments**.
3. A Vietnamese 2×2 dataset and protocol (real / sham / source / donor), with ≥ 5 test speakers.
4. A controlled analysis of whether models recognise "AI painting" or merely "processing traces".

**Part1 results (14 October; details in §15–§16).** The test set has only 3 speakers, so these are feasibility signals, not conclusions.
- Trained **without any generator fakes**, `tricue_mech` reaches **0.776** on MuseTalk (unseen). This is level with the supervised B-AVH at 0.800 (Δ −0.015 [−0.061; +0.035]), and **+0.075** [+0.050; +0.098] above the same architecture trained on Wav2Lip. On Wav2Lip it reaches 0.933 (−0.057 compared with training directly on Wav2Lip).
- Within the same partial clip, MechFake models **find the painted segment** of MuseTalk: the cell AUC inside vs outside the segment is 0.82, and the highest-scoring cell falls inside the segment in 88% of clips. Models trained on Wav2Lip reach only 0.65 and 42%.
- **They cannot yet separate "AI painting" from "processing"**:
  - every artifact-based model flags 79–100% of real clips whose mouth was merely pasted back (E6a);
  - they flag 52–78% of real clips with whole-frame processing (E6b);
  - only B-AVH passes (its false-alarm rate stays close to that on real clips).
- TriCue's "AI-painted?" output is effectively the artifact branch. The sync branch does its job on the "mismatch?" output (sham 0.943, MuseTalk donor 0.826). m_lead carries no signal.
- **No mechanism → generator map emerges.** Dropping M5, or using M4 only, gives the best MuseTalk scores, but those configurations were chosen after seeing the test set.

## 1. Motivation (from v001)

| Observation | v001 measurement | Consequence |
|---|---|---|
| The artifact branch memorises its generator | P2: Wav2Lip 0.979 → MuseTalk 0.553; artifact branch 0.98 → 0.64 | Training on one generator's fakes does not generalise |
| Strongest baseline | B-AVH: 0.837; MuseTalk 0.786; MuseTalk source 0.757 | The bar to beat |
| **Not yet shown that models "recognise AI"** | MuseTalk 0.786 is measured on MuseTalk fakes vs real + sham (no Wav2Lip mixed in), but the possibility that models only recognise the **crop–resize–paste trace** shared by all generators is not excluded | Needs a processing diagnostic (§5.4) |
| A→V reconstruction does not learn | R ≈ constant predictor in all 3 seeds | Sync branch uses an offset-classification head |
| Mismatch ≠ AI | Sync-only flags 88% of shams | Separate the two questions |
| Fakes are better synced than reals | Sync on W2L source: 0.436 | Needs evidence beyond sync (m_lead) |
| Cost | Fake generation ~12–13 h GPU per part (the largest share) | Generation can be dropped for training |

## 2. Problem and scope

The detector answers **two separate questions**:
1. **Was the mouth region AI-painted (regenerated)?**
2. **Are audio and mouth out of sync?**

Four cells (the audio is always a genuine recording):

| | Real visual | AI-painted visual |
|---|---|---|
| Original audio | Real | Fake source |
| Genuine audio spliced from elsewhere | Sham | Fake donor |

**Voice cloning** (adding an "AI-generated audio" row → 6 cells, a third question) is **deferred to v003**.

**Definition of "AI-painted" in v002:** the mouth region is **newly generated or regenerated** and inserted into the video. Benign whole-frame processing (re-encoding, whole-frame resizing, sharpening) does **not** count as AI.

## 3. Related work and gap

| Work | What it does | Gap relative to v002 |
|---|---|---|
| CutPaste (2021), DRAEM (2021) | Synthesise fake defects to learn real defects (industrial images) | Origin of the idea; not for video or lip-sync |
| SBI (Shiohara & Yamasaki, CVPR 2022) | Blend two variants of the same real image → fakes with paste boundaries and statistical mismatch | Whole face (face swap), one mechanism |
| UniForensics / DVSB (2024), SBV | Self-blended videos with temporal traces | Whole face, not per lip-sync mechanism |
| **SAVe (arXiv 2026)** | Real-only: blends two augmentations of the same frame over 3 regions (face, lips, lower half) + AVSync on AV-HuBERT | **One mechanism (blending)**, no temporal manipulation; the visual branch ≈ chance when only audio is fake; collapses at c40 compression (0.77) |
| **AVPF (arXiv 2026)** | Real-only: shifts video / audio by a few frames and blends within 0.5–1.5 s windows; splices one A/V frame from elsewhere in the same video; AV-HuBERT backbone | **No** downscale–upscale / VAE / warp; **no localisation**; no test on specific lip-sync generators reported |
| Rajan et al. (ICLR 2025) | Real ↔ real pairs reconstructed through an LDM autoencoder | For images; not yet seen for lip-sync mouths |
| AEROBLADE (CVPR 2024) | Autoencoder reconstruction error to detect latent-diffusion images, training-free | For images; used as a side check |

**Evidence on "learning from pseudo-fakes" vs "learning from generator fakes".** Numbers come from each paper under different conditions and are not directly comparable:

| Source | Result |
|---|---|
| SBI (CVPR 2022), trained on FF++ | Real + SBI only beats the best previous frame-level methods on CDF / DFDC / DFDCP by 6.08 / 5.17 / 5.23 points; cross-dataset average 87.33% vs 82.75% |
| Cheng et al. (NeurIPS 2024) | A SOTA detector trained only on blendfakes; **mixing deepfakes + blendfakes directly is worse than blendfakes alone ("1+1<2")** → needs a dedicated learning scheme (OPR) |
| Liu et al. (arXiv 2025, CS-DFD) | Joint training on real fakes + pseudo-fakes can hurt through gradient conflict |
| AVPF (2026) | Adding AVPF to AVH-Align when training with fakes: train FakeAVCeleb → test AV-Deepfake1M rises 57.7 → 73.2 AUC; TalkingHeadBench 77.8 vs 28.5 (AVH-Align) |
| SAVe (2026) | Real only: 0.96–0.97 on LipSyncTIMIT (original / c23) but 0.77 at c40 |

Overall reading:
- Pseudo-fakes usually **generalise better across datasets**, and may lose slightly in-domain.
- **Mixing them directly with real fakes can backfire**, so `tricue_mech_w2l` must be read carefully: if it is worse than `tricue_mech`, that matches "1+1<2" and is not a bug.
- There is no evidence yet on MuseTalk or on Vietnamese.

**Gap targeted by v002:** mouth-region pseudo-fakes **per generation mechanism** (M2–M4), a **mechanism → generator map**, **localisation of fake segments** without training on any generator, **Vietnamese**.

## 4. Method

### 4.1. MechFake — 5 mechanisms

Each real train/val clip yields the fake samples below. All parameters are **drawn at random from shared ranges**, not copied from the test generators (to reduce leakage, §8).

| Code | Mechanism imitated | Construction (proposed) | Typical generator | Literature |
|---|---|---|---|---|
| **M1 · Blend** | Pasting / compositing | Two augmentations (colour, sharpness, small shift) of the same mouth region, blended with a soft mask | All generators | Exists (SBI, SAVe) — kept as base |
| **M2 · Downscale–upscale** | Painting pixels at low resolution | Mouth / lower face → shrink to 30–70% of the true box side (random interpolation) → upscale → paste | Wav2Lip (96 px) and GAN family | Not found for lip-sync |
| **M3 · Through a VAE** | Decoding from a latent | Crop → SD-VAE ft-mse encode–decode, each latent cell covering 2–5 frame pixels (MuseTalk ≈ 3) → paste | MuseTalk, LatentSync | Not found for lip-sync (idea from Rajan 2025) |
| **M4 · Warp** | Warping a reference image | Thin-plate-spline warp of the mouth–jaw region to landmarks of another frame of the same clip (3–25 frames apart) | DINet, landmark-based family | Not found for lip-sync |
| **M5 · Temporal** | Wrong timing, jitter | (a) mouth taken from another time in the same clip; (b) hold / repeat 2–4 mouth frames; (c) mouth played 0.8–1.2× faster / slower | Footage splicing; temporally jittery generators | (a) close to AVPF; (b), (c) small, not found separately |

Common to all mechanisms:
- **Mask:** from landmarks (lips or lower face, chosen at random), elastically deformed with feathered edges (as in SBI / AVPF).
- **Full and partial:** partial is a contiguous segment of **random length** (proposed 0.2–3.0 s), recorded in `fake_intervals` like real fakes.
- **Random re-compression at training time** (CRF 18–35) for both reals and fakes, to avoid SAVe's collapse under strong compression. The final encode is the same as for all other samples.
- The audio is always the original audio → every MechFake is of the "source" kind (no mismatch), except M5 (a), which is mismatched.
- `generator = mech_m1 … mech_m5`, so the `by_generator` table separates them.

### 4.2. TriCue detector (trained without any generator)

```
audio ─┐
       ├─ AV-HuBERT ── offset head ─────────────── m_sync ─┐
lips ──┘                                                   │
lips + VAD ── lead–lag signature ───────────────── m_lead ─┼── 2×2 score fusion
mouth RGB ── DINOv2 ── painting-trace head ──────── r_art ─┘     ├─ mismatch?
                ↑ learns real vs MechFake (M1–M5)                 └─ AI-painted?
```

| Branch | Learns from | What it does |
|---|---|---|
| **m_sync** | Real + sham + self-shifted real (no fakes), then frozen | Classifies the audio–mouth offset over −15 … +15 frames plus a "no match" class (Synchformer idea). Replaces P2's stage A + stage S |
| **m_lead** | Real only | Asymmetric cross-correlation between lip aperture and the audio envelope around speech onsets (Chandrasekaran 2009; Schwartz & Savariaux 2014) |
| **r_art** | Real vs **MechFake** | Same architecture as P2's artifact branch (DINOv2 mouth crop 224, `[x ‖ Δx]`, TemporalBlock, 0.2 s cells) |
| **Fusion** | Real, sham, MechFake | Late fusion (no gate) into 2 outputs; the "mismatch" label comes from `av_mismatch_intervals` / shifts, the "AI-painted" label from MechFake |

**Train and validation contain no fakes from any generator.** Validation uses MechFake to select checkpoints. The main report uses threshold-free metrics (AUC, AP); threshold-dependent metrics state that the threshold was chosen on MechFake.

## 5. Experiments

### 5.1. Data and protocol

- **Part1 (feasibility):** ~~keep the old split~~ → **decision of 12 October:** regenerate part1 from scratch, keeping the clean clips.
  - Split **60/15/25** (re-exported), random-length partials, MechFake + diagnostics; caches re-extracted.
  - The test set is no longer the same as v001's, so B-AVH and P2 are retrained on the new part1 as reference points.
- **Part2 (main conclusions):**
  - 12–20 speakers, split **60/15/25** (test ≥ 5 speakers), random-length partials;
  - MechFake for train/val;
  - real generators **needed only for testing** (Wav2Lip + MuseTalk), unless the supervised baselines are kept (§10).
- **External test (optional, later):** English datasets with generators not used to design the mechanisms.
  - AVLips (on Hugging Face), FakeAVCeleb (registration required), LipSyncTIMIT (access unclear: Diff2Lip, VideoReTalking, IP-LAP, Wav2Lip).
  - This is a cross-language check, so results also mix in language effects.

### 5.2. Main comparison

Seeds 42/43/44, 20 epochs / patience 5. Architecture names are proposals, selected through the `ARCHITECTURES` list.

| Model | Trained on | Answers |
|---|---|---|
| `avh_tcn` (B-AVH) | Real + sham + Wav2Lip | Strongest supervised baseline (≈ AV-Lip-Sync+) |
| `p2_syncartifact` (P2) | Real + sham + Wav2Lip | v001 proposal |
| **`tricue_mech`** | Real + sham + **MechFake** | **v002 proposal** |
| `tricue_w2l` | Real + sham + Wav2Lip | Same architecture, old training |
| `tricue_mech_w2l` | Real + sham + MechFake + Wav2Lip | Do real fakes help? |
| `tricue_mech_no_lead` / `_no_sync` / `_no_art` | MechFake | Contribution of each branch |
| ~~`tricue_mech_one_head`~~ | — | **Dropped at implementation (§14).** TriCue learns each branch, then fuses. Its "AI-painted?" head already learns sham = 0 like a single-output model, so the variant no longer differs. The question "is sham called AI?" is read directly from the AI head's sham FAR and the `mismatch` head's sham AUC |

v001 numbers are kept as references without retraining: B-FATE, `avh_realrecon`, and the 4 P2 variants.

### 5.3. Mechanism analysis (r_art branch only, cheap)

| Run | Purpose |
|---|---|
| **M1 only** (≈ SAVe / SBI) | Reproduces published blending as a baseline |
| **M5 only** (≈ the splicing part of AVPF; M5 includes a/b/c in one generator) | Reproduces published temporal splicing |
| **M1 + M5 only** (≈ SAVe + AVPF) | Strongest combination of published approaches; reference to measure the value of M2–M4 |
| **All of M1–M5** (`art_all`, added at implementation) | Reference point for every analysis configuration (all are "painting-trace branch only") |
| **Drop M1 … M5 one at a time** | **Mechanism → generator map**: does MuseTalk drop without M3? does Wav2Lip drop without M2? |
| **One mechanism only** (M2, M3, M4 alone) | How much signal each new mechanism carries by itself |

### 5.4. "AI or processing" diagnostics

| Code | Test | Applies to | Expected if the model is right |
|---|---|---|---|
| **E6a** | Real mouth → crop → resize → paste back, labelled real | Models trained on **Wav2Lip** (B-AVH, P2, `tricue_w2l`) | Not flagged as AI. If it is → "recognises MuseTalk 0.79" actually means "recognises the paste trace" |
| **E6b** | Benign **whole-frame** processing (re-compress CRF 28–35, whole-frame resize, sharpening / filters), labelled real | **All models** | FAR ≈ real FAR |
| **E7** | Localisation on MuseTalk partials only: do temporal scores rise inside the painted segment? | All models (from `predictions-test.json`) | Yes. If scores are uniformly high across the clip → the model looks at clip-level properties, not the painted part |

E6a does **not** apply to MechFake models, because M2 teaches exactly that operation as "AI-painted" under the §2 definition. For MechFake, the corresponding test is E6b.

### 5.5. Metrics

As in v001:
- ROC-AUC, AP, F1, false-alarm rate, temporal AP;
- breakdowns by generator, 2×2 cell and branch;
- paired ΔAUC with bootstrap over source clips.

Added in v002:
- separate metrics for the 2 outputs;
- a per-mechanism table (`mech_m*`);
- FAR on E6a / E6b.

## 6. Gates and roadmap

| Gate | Work | Cost (estimate) | Proceed if |
|---|---|---|---|
| **G0** ✔ | Read AVPF (12/10/2026) | — | Done: no overlap with M2–M4, the mechanism map or localisation; M5 (a) is close to AVPF → reported as a reproduction |
| **G1** | Build M1, M2, M5 (CPU) + M3 (Kaggle); linear DINOv2 probe real vs MechFake → test on part1 Wav2Lip and MuseTalk | 1–2 days + ~1–2 h GPU | **Both > 0.65** and MuseTalk > 0.64 (P2 artifact branch) |
| **G2** | Add M4 (dlib landmarks on CPU, ~1–1.5 h); mechanism analysis §5.3 | ~1 h GPU | A mechanism explains a generator |
| **G3** | m_sync, m_lead (E1: lead–lag signature, CPU); assemble TriCue; comparison §5.2; diagnostics §5.4 on part1 | ~3–5 h GPU | `tricue_mech` ≥ `tricue_w2l` on MuseTalk, without falling too far on Wav2Lip |
| **G4** | Part2: collect, review, export 60/15/25, generate test, MechFake; rerun everything | Test generation ~half of v001's generation cost (estimate) + ~4–6 h | Main conclusions |
| **G5** (optional) | External test (§5.1) | Depends on access | — |

Failing G1 → stop MechFake (~2 days lost) and return to training the artifact branch on Wav2Lip, as planned on 11 October.

**Number of training runs:**
- 8 full models (§5.2, without `one_head`);
- 12 r_art-only configurations (§5.3: `art_all` + 6 "only" + 5 "drop one");
- **20 configurations × 3 seeds = 60 runs**.

**Part1 time estimate.** Derived from v001: 24 runs = 2.8 h GPU; DINOv2 0.4 h / 2954 samples; AV-HuBERT 4.7 h / 2954 samples. MechFake ≈ 5 samples per real train + val clip ≈ 2500 samples.

| Item | Where | Time (estimate) |
|---|---|---|
| Part1 dlib landmarks | Local CPU | ~1–1.5 h |
| Build M1, M2, M4, M5 + encode; E6a / E6b data | Local CPU | ~2–4 h |
| Build M3 (SD-VAE) | GPU | ~0.5–1 h |
| DINOv2 for MechFake + E6 (mouth box from the original real clip) | GPU | ~0.4–0.5 h |
| AV-HuBERT for **a subset** of MechFake (~1 sample / clip, enough to train the fusion head) | GPU | ~0.8 h (all 2500 samples: ~4 h) |
| G1 probe | GPU | < 0.2 h |
| Train 9 full models × 3 seeds (~10 min / run) | GPU | ~4.5 h |
| Train 11 r_art configurations × 3 seeds (~7 min / run) | GPU | ~3.9 h (1 seed: ~1.3 h) |
| **Total GPU** | | **~10–12 h** (~1/3 of the weekly quota); ~8–9 h if the mechanism analysis uses 1 seed |

## 7. Expected results (hypotheses)

- `tricue_mech` drops on Wav2Lip relative to 0.98 (no longer trained on it) but rises on MuseTalk relative to 0.55–0.64.
- **Success** = both Wav2Lip and MuseTalk at a decent level (e.g. ≥ 0.8) without training on any generator, approaching or beating B-AVH (0.786 on MuseTalk).
- Dropping M3 hurts MuseTalk and dropping M2 hurts Wav2Lip → the mechanism map is meaningful.
- E6b: FAR ≈ real. E7: scores rise inside the painted segment.
- Note: the very high SAVe / AVPF scores are measured on FakeAVCeleb, which is known to contain shortcuts. MuseTalk is harder, so similar numbers are not expected.

**A minimal result is still valuable** even if numbers are not high: the mechanism → generator map, the Vietnamese dataset + protocol, and a controlled answer to "AI or processing".

## 8. Risks and mitigations

| Risk | Level | Mitigation |
|---|---|---|
| **Leaking test-generator knowledge**: knowing Wav2Lip paints at 96 px and MuseTalk uses SD-VAE | High (methodological) | Wide random parameter ranges; state the protocol clearly as "mechanism known, generator unseen"; external test on generators not used for the design |
| Pseudo-fakes too easy, the model memorises them | Medium | Soft masks, several intensity levels, re-compression; monitor with leave-one-mechanism-out |
| Drop on Wav2Lip | Almost certain | Accepted; `tricue_mech_w2l` as comparison |
| Threshold chosen on MechFake mismatched to real fakes | Medium | Main report uses AUC / AP |
| VAE fingerprint erased by pasting, compression, upscaling to 224 | Medium | Try native-resolution crops / high-frequency features before dropping M3 |
| E1 (m_lead) carries no signal (25 fps = 40 ms) | High | Drop m_lead, TriCue keeps 2 branches |
| Hot area (SAVe in March, AVPF in April 2026) | Medium | Anchor on M2–M4, the mechanism map, localisation, Vietnamese |
| Few test speakers in part1 (after re-export to 60/15/25: **3 speakers**) | As for any direction | Part1 = feasibility signal; conclusions on part2 |

## 9. Decisions taken

| Decision | Date |
|---|---|
| MechFake direction (train without generator fakes) | 12/10 |
| Keep 2 real-only branches (m_sync, m_lead) + 2 outputs; m_sync frozen; **no** variant where m_sync sees fakes | 11/10 |
| Drop B-FATE (no PE-AV + FATE extraction); v001 numbers kept as reference | 11/10 |
| Random-length partials | 11/10 |
| Split 60/15/25 from part2; part1 keeps the old split | 11/10 |
| Feasibility first on all of part1 | 11/10 |
| Voice cloning (6 cells) deferred to v003 | 11/10 |
| New encoder / generator suggestions (CLIP, MARLIN, BRAVEn, XLAVS-R, LatentSync) deferred unless needed | 11/10 |

## 10. Keep / replace / drop / add / defer relative to v001

| | Content |
|---|---|
| **Keep** | Pipeline 01→05, clip-cutting rules, manual review, connected-group split; 2×2, sham AI = 0; AV-HuBERT + DINOv2 encoders; MuseTalk + Wav2Lip at test; evaluation framework (`by_condition`, `branches`, paired ΔAUC, test-lock, `compare.ipynb`), W&B, Kaggle notebooks |
| **Replace** | Wav2Lip fakes at training time → MechFake; stage A + S → offset head; gate → late fusion with 2 outputs |
| **Drop** | B-FATE; `avh_realrecon` and the 4 P2 variants are not rerun (code kept) |
| **Add** | MechFake generation step; m_lead branch + lip-aperture / speech-onset features; diagnostics E6a / E6b / E7; AV-HuBERT worker saves landmarks (part2) |
| **Defer** | LatentSync as a second held-out generator; other encoders; generator / speaker gradient reversal; real-time, advanced localisation, person-of-interest, heart rate; voice cloning (v003) |

## 11. Code changes (original plan; what was done is in §14)

| Part | Change |
|---|---|
| New: MechFake generation step (in `generation` or a separate script) | M1–M5, landmark masks, random partials, re-compression, correct label schema; M3 uses the SD-VAE from the MuseTalk environment |
| `generation.py` | Random-length partials; separate real / sham from generator jobs (part2 only generates test generators if the supervised baselines are dropped). Changes the signature → old outputs cannot be resumed |
| `avhubert_worker.py` | Save 68 landmarks (part2; changes the cache signature) |
| Part1 script | dlib landmarks on CPU + VAD onsets (E1, M4), without touching the old cache |
| `split.py` / `data_settings.py` | 60/15/25 for part2 |
| `models.py` / `experiment.py` | `tricue_*`, offset head, m_lead, 2 outputs, late fusion |
| `experiment.evaluate`, `compare.ipynb` | 2 outputs, per-mechanism table, E6a / E6b / E7 |

## 12. Approval decisions (12/10/2026)

| Item | Decision |
|---|---|
| MechFake + TriCue proposal | **Approved**, implemented (§14) |
| Wav2Lip in train/val | **Kept** (supervised baselines B-AVH, P2, `tricue_w2l`) |
| Boldisor baseline (training-free sync) | Not done |
| Score calibration | Not done |
| More sham types | Not done |
| Clips per speaker limit | **No limit**; `MAX_CLIPS_PER_SPEAKER` available to enable later |
| Part1 split | Recomputed to **60/15/25** (re-exported, clips unchanged); part1 regenerated from scratch |
| Mouth box for MechFake DINOv2 | Taken from the original real clip (`box_sample_id`) |
| External test (G5) | Undecided |
| Environment | Main env on **Python 3.11** (AV-HuBERT / MuseTalk workers stay on 3.10) |
| Old data | Deleted `runs/train_part1_run01`, all caches, FATE weights/sources, the generated part1 (backed up); `data_pipeline/` kept unchanged |

## 13. Key references

- Shiohara, Yamasaki. *Detecting Deepfakes with Self-Blended Images.* CVPR 2022. arXiv 2204.08376
- Cheng et al. *Can We Leave Deepfake Data Behind in Training Deepfake Detector?* NeurIPS 2024. arXiv 2408.17052
- Liu et al. *Suppressing Gradient Conflict for Generalizable Deepfake Detection.* arXiv 2507.21530, 2025
- Shahzad et al. *SAVe: Self-Supervised Audio-visual Deepfake Detection Exploiting Visual Artifacts and Audio-visual Misalignment.* arXiv 2603.25140, 2026
- Wei, Li. *Generalizing Video DeepFake Detection by Self-generated Audio-Visual Pseudo-Fakes (AVPF).* arXiv 2604.09110, 2026
- *UniForensics: Face Forgery Detection via General Facial Representation.* arXiv 2407.19079, 2024
- Li et al. *CutPaste: Self-Supervised Learning for Anomaly Detection and Localization.* arXiv 2104.04015, 2021
- Rajan, Ojha, Schloesser, Lee. *Aligned Datasets Improve Detection of Latent Diffusion-Generated Images.* ICLR 2025. arXiv 2410.11835
- Ricker, Lukovnikov, Fischer. *AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error.* CVPR 2024. arXiv 2401.17879
- Zhang et al. *DINet: Deformation Inpainting Network for Realistic Face Visually Dubbing on High Resolution Video.* AAAI 2023. arXiv 2303.03988
- Prajwal et al. *Wav2Lip.* ACM MM 2020; Zhang et al. *MuseTalk.* arXiv 2410.10122
- Iashin et al. *Synchformer.* ICASSP 2024. arXiv 2401.16423
- Chandrasekaran et al. *The Natural Statistics of Audiovisual Speech.* PLoS Comput Biol 2009; Schwartz, Savariaux. PLoS Comput Biol 2014
- Smeu et al. *AVH-Align.* CVPR 2025; Boldisor et al. CVPR 2026. arXiv 2511.17181
- Liu et al. *LipFD / AVLips.* NeurIPS 2024; Datta et al. *LIPINC-V2 / LipSyncTIMIT.* arXiv 2504.01470

## 14. Code implementation (12/10/2026)

Repository `vn-av-df-forensics`, on top of commit `4c239de`. Tests: **117 passed, 1 skipped** (symlinks on Windows); ruff clean. Tests use fake generators and fixture caches; they **do not prove** that MechFake / TriCue run on real data and GPUs.

| Part | Change |
|---|---|
| Removed | FATE (`features/fate.py`, `features/face.py`, PE-AV/FATE downloads), GRU head, `avh_realrecon` (`RealReconDetector`), the 4 P2 variants; dropped `peft`, `accelerate`, `timm`, `torchvision` from `[training]` |
| `mechfake.py` (new) | M1 `mech_blend`; M2 `mech_lowres` (shrink to 30–70% of the mouth-box side); M3 `mech_vae` (SD-VAE ft-mse, latent cell 2–5 frame px); M4 `mech_warp` (DIS optical flow, frames ±3–25 apart); M5 `mech_temporal` (offset ±5–25 / hold 2–4 / speed 0.8–1.2); `diag_paste` (E6a); `diag_global` (E6b: downscale / sharpen / gamma, then re-encode at CRF 28–35). Mouth region from YuNet (mouth corners); deformed, feathered elliptical mask fixed per clip. Face missing in > 20% of frames → pair dropped, as for real generators |
| `generation.py` | **Random-length** partials `partial_range_s` (default 0.2–3.0 s); MechFake / diagnostics are generators in the same plan (original audio, no worker); one mechanism per parent with `feature_scope = full` (AV-HuBERT extracted to train fusion), the rest `artifact` + `box_sample_id`; diagnostics labelled 0, `diagnostic`, test only; MechFake of parents that lost their real are dropped |
| `features/avhubert_worker.py` | Cache **v2** additionally stores 68 landmarks and the audio envelope (RMS / 40 ms) |
| `features/dinov2.py` | Samples without an AV-HuBERT cache borrow the mouth box of the original real clip (`box_sample_id`) |
| `tricue.py` (new) | m_sync (cosine profile ±K → TCN offset classifier + "no match"), m_lead (Mahalanobis on the lead–lag profile, not trained), r_art (DINOv2 → TCN), non-negative logistic fusion → 2 outputs. Trained stage by stage: statistics → S → L → R → F |
| `experiment.py` | Filters train/val fakes by each method's `train_generators` (glob, e.g. `mech_*`); samples with artifact cache only; stage-wise training for `tricue` / `artifact`; evaluation adds `diagnostics` (E6) and `localization_check` (E7); checkpoint format `v2` |
| `settings.py`, notebooks | 8 main methods + 12 mechanism-analysis configurations; generate gains `MECHANISMS`, `PARTIAL_RANGE_S`; prepare only AV-HuBERT + DINOv2; train gains `TRICUE`, `LEAD`, `fusion_*`; compare uses v002 names/pairs |
| Data pipeline | VAD silence merging `merge_silence_seconds` 0.6 s; **face check on every frame** of kept windows (`every_frame_face`); split 60/15/25 + group counts per split + warning when test < 5 groups; `MAX_CLIPS_PER_SPEAKER` (off) |
| Python | `requires-python >= 3.11` (both packages), ruff `py311` |

**Differences from the proposal, and why:**
- **Branch balancing (B2) dropped:** branches are trained stage by stage and fusion is just a few non-negative weights, so there is no gate that one branch could dominate.
- **`tricue_mech_one_head` dropped:** see §5.2.
- **Added `art_all`:** the all-5-mechanisms reference for the analysis.
- **M4 uses optical flow, not lip landmarks:** so MechFake does not depend on dlib (absent from the main env).
- **m_lead finds speech onsets from the audio envelope**, not Silero VAD.
- **M2/M3 scale with the true mouth-box side** (fixed after the 12/10 trial run):
  - generation frames are 640 px, so mouth boxes are only ~50–120 px;
  - the previous fixed 48–160 px side (M2) was larger than the box, so downscale–upscale lost nothing and the fakes were empty (measured difference ≈ 0);
  - `diag_paste` keeps 96 px like Wav2Lip (at 640 px almost only the paste boundary remains).
- **M5 uses the lip region only** (same fix round):
  - taking the lower face from another frame dragged along neck/clothing in a different pose, creating a large paste trace (overlapping M1); M5 must carry only the wrong-timing signal;
  - visual check on 3 part1 clips: M1/M2/M4/M5 clearly visible, M3 almost invisible (a decoder fingerprint, as Rajan 2025 suggests).
- **Generation speed:** multi-threaded PyAV decoding (bit-identical frames), and the 5 mechanisms of a parent share the decoded clip and face points. The MechFake part per pair went from ~13 s to ~6 s (local).

**Next steps (runs) — done 12–14/10:**
1. ✔ Reinstall the main env on Python 3.11.
2. ✔ Re-run `05_export` for part1 (60/15/25).
3. ✔ Upload `vn-av-df-clean-part1`.
4. ✔ `generate.ipynb` in 3 independent sessions (Wav2Lip on GPU ‖ MechFake + diagnostics on CPU ‖ MuseTalk + `mech_vae` on GPU), merged locally with `generation/03b_merge.py`.
5. ✔ Review / finalize / export.
6. ✔ `prepare.ipynb` (AV-HuBERT v2 for 4444 samples + DINOv2 for 7892 samples).
7. ✔ No separate G1 probe; `art_all` was used directly (§15.5).
8. ✔ `train.ipynb` for all 20 configurations, one run per seed (`train_part1_v002_s42/_s43/_s44`); the test was run on a local CPU (§15.1).

**Fixes after the runs (13–14/10):**
- MuseTalk ran out of GPU memory after `mech_vae` (batch 16 with patches up to ~600 px). Fix: VAE batch size scales with patch size, and the CUDA cache is released.
- `reporting.py` raised `KeyError: 'train_loss'` for stage-trained models (history has a single row), so the notebook stopped before the test cell in all 3 runs. Fix: plot per-stage losses from `stages.json` and record `stage_epochs`; the test was rerun locally.
- **Differs from the proposal, not done:** random re-compression at CRF 18–35 during training (§4.1) was **not implemented**. This bears directly on the E6b result (§15.6).

---

## 15. Part1 results (14/10/2026)

Numbers are **measured on the v002 part1 test set** and are not directly comparable with v001's test (different split). Values are mean ± std over seeds 42/43/44; "ensemble" averages the scores of the 3 seeds.

The CIs in §15.3 are bootstraps over **source clips** (140 clips), so they are **optimistic**: the test set has only 3 speakers, and the code needs at least 5 groups to compute per-speaker CIs.

Full tables: `reports/compare/test_vn-av-df-data-part1.md` (+ PNG) in the archive; the v001 version is in `reports/compare/v001_part1/`.

### 15.1. Experiment record

| Field | Value |
|---|---|
| Run dates | Generation 12–13/10 (3 independent sessions, merged locally); prepare 13/10; training 13–14/10 (3 Kaggle sessions, one per seed); test 14/10 locally (CPU) |
| Runs | `runs/train_part1_v002_s42`, `_s43`, `_s44`; same manifest `5a1da22d…`; each run has a `test-lock.json` |
| Models | 20 configurations (§5.2 + §5.3) × 3 seeds = 60 detectors |
| Why the test ran locally | The validation report cell failed (§14), stopping the notebook before the test cell. The test was rerun from the downloaded checkpoints and caches, under the same protocol: thresholds and checkpoints locked on validation, run once |
| Generators | Wav2Lip GAN; MuseTalk 1.5 (test only); MechFake M1–M5; diagnostics `diag_paste` (E6a) and `diag_global` (E6b) |
| Encoders | AV-HuBERT Base LRS3 iter4 (cache v2, with landmarks and audio envelope); DINOv2-small on 224 mouth crops |

### 15.2. Data

| Split | Speakers | Source clips | Samples | Composition | Hours |
|---|---|---|---|---|---|
| Train | 4 (05, 06, 07, 09) | 342 | 4788 | 342 real, 342 sham; 684 W2L donor (full/partial); 684 per MechFake mechanism | 10.19 |
| Validation | 3 (02, 03, 11) | 89 | 1424 | 89 real, 89 sham; 356 W2L (donor/source × full/partial); 178 per mechanism | 2.85 |
| Test | **3 (01, 08, 10)** | 140 | 1680 | 140 each of real, sham, `diag_paste`, `diag_global`; 560 W2L; 560 MuseTalk | 3.38 |

- **7,892 samples / 16.4 hours** in total, from 575 clean clips. 571 are usable: 4 Wav2Lip pairs with no detectable face were dropped together with their source clip.
- Partials last 0.2–3.0 s (median 1.56 s); full : partial = 1 : 1.
- AV-HuBERT was extracted for 4,444 samples with `feature_scope = full`: real, sham, real generators, and one MechFake mechanism per source clip. DINOv2 was extracted for all 7,892.
- **Each model has its own validation set**, containing only the generators it was trained on (N from 356 to 1068). Validation numbers are therefore **not comparable across models**; all comparisons use the test set.
- On the 640 px frames, mouth boxes are only **45–85 px** (per-speaker median), upscaled to 224 for DINOv2.

### 15.3. Video-level detection on test

| Model | Trained on | ROC-AUC | Ensemble | Wav2Lip | **MuseTalk** | FAR real / sham |
|---|---|---|---|---|---|---|
| B-AVH `avh_tcn` | W2L | 0.846 ± 0.008 | 0.846 | 0.892 | **0.800** ± 0.012 | 0.29 / 0.30 |
| P2 `p2_syncartifact` | W2L | 0.838 ± 0.025 | 0.860 | **0.990** | 0.687 ± 0.048 | 0.12 / 0.12 |
| `tricue_w2l` | W2L | 0.839 ± 0.019 | 0.851 | **0.990** | 0.689 ± 0.039 | 0.15 / 0.15 |
| **`tricue_mech`** | **MechFake** | 0.855 ± 0.035 | 0.860 | 0.933 | **0.776** ± 0.053 | 0.17 / 0.18 |
| `tricue_mech_w2l` | MechFake + W2L | 0.855 ± 0.043 | 0.856 | 0.947 | 0.763 ± 0.068 | 0.33 / 0.34 |
| `tricue_mech_no_lead` | MechFake | 0.854 ± 0.036 | 0.859 | 0.932 | 0.776 | 0.17 / 0.18 |
| `tricue_mech_no_sync` | MechFake | 0.843 ± 0.022 | 0.846 | 0.922 | 0.764 | 0.18 / 0.17 |
| `tricue_mech_no_art` | MechFake | 0.522 ± 0.004 | 0.522 | 0.519 | 0.526 | 0.82 / 0.96 |

By 2×2 cell:
- Artifact-based models have **donor ≈ source**. For example, `tricue_mech`: MuseTalk 0.796 / 0.778; Wav2Lip 0.940 / 0.932.
- B-AVH has **donor > source**, as in v001: MuseTalk 0.839 / 0.778; Wav2Lip 0.939 / 0.857.

**Paired ΔAUC (ensemble, 2000 bootstrap resamples over source clips):**

| Comparison | Overall | Wav2Lip | MuseTalk |
|---|---|---|---|
| TriCue-Mech − TriCue-W2L | +0.009 [−0.010; +0.029] | **−0.057** [−0.077; −0.037] | **+0.075** [+0.050; +0.098] |
| TriCue-Mech − B-AVH | +0.014 [−0.024; +0.051] | +0.042 [+0.013; +0.070] | −0.015 [−0.061; +0.035] |
| TriCue-Mech − P2 | −0.000 [−0.021; +0.021] | −0.056 [−0.076; −0.036] | +0.056 [+0.028; +0.090] |
| TriCue-Mech+W2L − TriCue-Mech | −0.004 [−0.012; +0.004] | +0.016 [+0.009; +0.023] | **−0.024** [−0.036; −0.011] |
| TriCue-Mech − TriCue −lead | +0.001 [−0.000; +0.002] | +0.000 | +0.001 |
| TriCue-Mech − TriCue −sync | +0.014 [+0.010; +0.017] | +0.006 | +0.021 [+0.017; +0.026] |
| TriCue-Mech − TriCue −art | +0.338 [+0.303; +0.376] | +0.416 | +0.259 |

### 15.4. Per speaker

Ensemble AUC, computed on each generator's fakes vs label-0 samples of the same speaker.

| Model | 01 W2L | 01 MT | 08 W2L | 08 MT | 10 W2L | 10 MT |
|---|---|---|---|---|---|---|
| B-AVH | 0.908 | **0.850** | 0.917 | 0.860 | 0.925 | 0.830 |
| P2 | 0.984 | 0.787 | 1.000 | 0.618 | 1.000 | 0.923 |
| TriCue-W2L | 0.983 | **0.797** | 1.000 | 0.811 | 1.000 | 0.911 |
| TriCue-Mech | 0.938 | 0.745 | 0.997 | **0.967** | 0.999 | **0.963** |
| TriCue-Mech+W2L | 0.938 | 0.711 | 1.000 | 0.984 | 0.999 | 0.978 |
| r_art M1–M5 | 0.940 | 0.702 | 0.997 | 0.978 | 0.996 | 0.952 |
| r_art M4 only | 0.972 | 0.775 | 0.999 | 0.988 | 0.998 | 0.968 |

- **Within a speaker**, MechFake models separate MuseTalk almost perfectly for 2 of 3 speakers (0.95–0.99). The pooled AUC is only ~0.78 because **baseline scores differ between speakers**. For example, the mean real score of `tricue_mech` is 0.73 for speaker 01, 0.03 for speaker 08 and 0.13 for speaker 10.
- **Speaker 01 reverses the result**: MechFake is below W2L (0.745 vs 0.797) and below B-AVH (0.850). So "MechFake beats W2L on MuseTalk" holds for **2 of 3 speakers**.
- Speaker 01 has 2 videos at 1080p with a bitrate of 1.8–2.0 Mbps (speaker 08: 0.63, speaker 10: 0.89) and a mouth box of ~49 px (like speaker 08). **It is not yet clear why** speaker 01's real videos score high. B-AVH does not show this for speaker 01, but does for speaker 10 (real score 0.59).

### 15.5. Mechanism analysis (r_art branch only)

| Configuration | Overall | Wav2Lip | MuseTalk | Overall Δ vs `art_all` (CI excludes 0?) |
|---|---|---|---|---|
| `art_all` (M1–M5) | 0.840 ± 0.019 | 0.918 | 0.762 | — |
| M1 only (≈ SBI/SAVe) | 0.836 | 0.940 | 0.732 | +0.000 |
| M2 only (downscale–upscale) | 0.844 ± 0.051 | **0.955** | 0.733 ± 0.090 | |
| M3 only (VAE) | 0.861 | 0.938 | 0.784 | |
| M4 only (warp) | **0.877** | 0.939 | **0.816** | |
| M5 only (temporal, ≈ AVPF) | 0.826 ± 0.030 | 0.935 | 0.718 ± 0.067 | +0.012 (no) |
| M1 + M5 (≈ SAVe + AVPF) | 0.836 | 0.931 | 0.742 | +0.003 (no); MuseTalk +0.026 ✓ |
| Drop M1 | 0.854 | 0.931 | 0.777 | −0.023 ✓ |
| Drop M2 | 0.868 | 0.947 | 0.789 | −0.026 ✓ |
| Drop M3 | 0.855 | 0.937 | 0.774 | −0.009 ✓ (MuseTalk −0.003, no) |
| Drop M4 | 0.868 | 0.941 | 0.796 | −0.013 ✓ |
| Drop M5 | **0.883** | 0.946 | **0.819** | −0.037 ✓ |

A negative Δ means `art_all` is **worse** than that configuration.
- **Dropping any single mechanism is as good as or better than using all 5.**
- The hypothesis "dropping M3 hurts MuseTalk, dropping M2 hurts Wav2Lip" is **not supported**. Dropping M3 changes MuseTalk by −0.003, not significant; dropping M2 even raises Wav2Lip.
- One weak signal points the expected way: M2 alone gives the highest Wav2Lip score (0.955) and the worst localisation (AP@0.5 = 0.21), consistent with downscale–upscale resembling Wav2Lip's 96 px painting. The margin over the other mechanisms is only ~0.015.

### 15.6. "AI painting" or "processing" (E6) and in-clip localisation (E7)

| Model | FAR real | **E6a: mouth pasted back** | **E6b: whole-frame processing** | E7 W2L (cell AUC) | E7 MT (cell AUC) | E7 MT: top cell inside segment |
|---|---|---|---|---|---|---|
| B-AVH | 0.29 | **0.23** | **0.26** | 0.88 | 0.82 | 0.73 |
| P2 | 0.12 | 0.89 | 0.65 | 0.99 | 0.69 | 0.27 |
| TriCue-W2L | 0.15 | 0.95 | 0.66 | 1.00 | 0.65 | 0.42 |
| TriCue-Mech | 0.17 | 0.86 | 0.54 | 0.95 | **0.82** | **0.88** |
| TriCue-Mech+W2L | 0.33 | 0.99 | 0.66 | 0.96 | 0.78 | 0.86 |
| r_art M1–M5 | 0.33 | 0.89 | 0.62 | 0.94 | 0.80 | 0.83 |
| r_art M2 only | 0.06 | 0.79 | 0.38 | 0.95 | 0.70 | 0.47 |
| r_art M4 only | 0.18 | 0.79 | 0.54 | 0.94 | **0.85** | 0.85 |
| r_art M1 + M5 | 0.57 | 1.00 | 0.78 | 0.95 | 0.78 | 0.85 |

FAR is the share of label-0 clips flagged as AI at the threshold chosen on validation. E7 uses fake partials only: it compares cells inside and outside the painted segment **of the same clip**, so clip-level properties cannot help.

- **E6a:** models trained on Wav2Lip flag 89–95% of real clips that were only cropped, resized and pasted back. Part of their "recognises MuseTalk" ability (0.69) is therefore **recognising the paste trace**. B-AVH passes (0.23, close to its real FAR of 0.29).
- **E6b:** every artifact-based model flags 38–78% of real clips that were only downscaled, sharpened or gamma-adjusted and re-encoded at CRF 28–35, well above their FAR on real clips. B-AVH stays close to its real FAR. The artifact branch, whether trained on MechFake or on Wav2Lip, is learning **"processing traces"**, not specifically "AI painting". Random re-compression during training (§4.1) was not implemented (§14).
- **E7:** on MuseTalk, MechFake models **put their highest score inside the painted segment** in 88% of clips, versus 27–42% for models trained on Wav2Lip. With an unseen generator, MechFake looks at **the painted part**, not at clip-level properties.

### 15.7. TriCue's two outputs and its branches

Threshold-free AUC, each cell vs original real.

| Output / branch | MT donor | MT source | W2L donor | W2L source | Sham |
|---|---|---|---|---|---|
| **"Mismatch?"** (`mismatch`, TriCue-Mech) | **0.826** | 0.670 | 0.767 | 0.526 | **0.943** |
| m_sync (TriCue-Mech) | 0.826 | 0.670 | 0.766 | 0.525 | 0.943 |
| m_lead | 0.571 | 0.531 | 0.599 | 0.567 | 0.493 |
| r_art (TriCue-Mech) | 0.776 | 0.768 | 0.931 | 0.928 | 0.500 |
| P2 / sync (reference) | 0.845 | 0.750 | 0.756 | 0.592 | 0.938 |

- **The "AI-painted?" output is effectively r_art.** Fusion weights (seed 42) are sync 0.10 / lead 0.18 / artifact 0.66 for `tricue_mech`, and 0 / 0 / 1.06 for `tricue_w2l`. The reasons:
  - shams (AI label 0) have the highest sync scores;
  - MechFake keeps the original audio;
  - Wav2Lip paints lips that match the audio.
  Under the current label definition, the sync branch therefore carries no "AI" information.
- AUC 0.52 without the artifact branch is a **genuine result**, not a bug. Its FAR of 0.89 comes from the max-F1 threshold degenerating into "flag everything".
- **The "mismatch?" output does its job.** It detects shams (0.943) and donor fakes **of the unseen generator** too (MuseTalk donor 0.826), and barely reacts to audio-matched source fakes (Wav2Lip source 0.526). It is robust to the generator.
- **m_lead carries no signal** in any cell (0.49–0.60), as the risk in §8 anticipated (a lead–lag signature at 25 fps).
- **A methodological flaw found during diagnosis:** stage F learns the fusion weights on **the same training data** the branches were trained on. Branch scores are therefore better on train than on unseen data:
  - the artifact score of reals is 0.16 on train vs 0.53 on validation;
  - the sync score is 0.09 vs 0.38.
  Calibration is shifted as a result: stage F's validation loss (1.35–1.8) is well above its train loss (0.79).
- Check: fitting the weights on half of validation and measuring on the other half, repeated 100 times. Fused AUC vs artifact only is 0.888 vs 0.889 (`tricue_mech`) and 0.939 vs 0.942 (`tricue_mech_w2l`). The flaw **does not change the conclusions**; nothing was retrained, and it is to be fixed in v003 (§19).

### 15.8. Temporal localisation (AP by tIoU, test)

| Model | AP@0.3 | AP@0.5 | AP@0.75 |
|---|---|---|---|
| B-AVH | 0.586 | 0.434 | 0.225 |
| P2 | 0.523 | 0.460 | 0.381 |
| TriCue-W2L | 0.524 | 0.477 | 0.399 |
| TriCue-Mech | 0.430 ± 0.175 | 0.360 ± 0.144 | 0.262 |
| r_art drop M2 | **0.669** | **0.575** | **0.455** |
| r_art M1 + M5 | 0.652 | 0.518 | 0.416 |

Whole-set AP uses one threshold for every clip. For MechFake models, the per-speaker baseline shift (§15.4) pulls AP down and makes it vary strongly across seeds. To answer "does the model look at the painted segment?", E7 (within-clip comparison) is the cleaner measure.

### 15.9. Measured cost

| Item | Where | Amount | Time |
|---|---|---|---|
| Wav2Lip generation | Kaggle 2×T4 | 804 pairs | 5.65 h (50.6 s/pair) |
| MechFake generation (except VAE) + diagnostics | Kaggle **CPU** (2 workers) | 2020 pairs | 8.37 h (24–35 s/pair), no GPU quota used |
| MuseTalk + `mech_vae` generation | Kaggle 2×T4 | 715 pairs | 6.27 h (MuseTalk 83.7 s/pair, VAE 49.9 s/pair) |
| AV-HuBERT extraction | 2×T4, 2 workers/GPU | 4444 samples | 8.04 h |
| DINOv2 extraction | 2×T4 | 7892 samples | 1.01 h |
| Training 20 configurations | Kaggle GPU, 1 session/seed | 60 detectors | 5.1 h in total (~1.7 h/seed) |
| Test | Local CPU | 60 detectors × 1680 samples | ~18 min/seed |
| **Total GPU** | | | **~26 h ≈ one week of quota** (+ 8.4 h CPU) |

- The estimate in §6 (10–12 h GPU) was **about 2× too low**. It left out Wav2Lip and MuseTalk generation for the baselines and the test set, and it assumed AV-HuBERT would be extracted only for a small subset, whereas 4,444 samples were extracted.
- Training is cheap: 2–14 min per configuration per seed. Experiments on **existing caches** are almost free; the real cost lies in generation and extraction.
- `resources.json` records 3–7 "trainable parameters" for TriCue, because it only counts the fusion weights still trainable at the end. This is a bookkeeping error and does not affect the results.

## 16. Discussion

### 16.1. By v002 question

| Question | Answer on part1 | Confidence |
|---|---|---|
| Does training without any generator generalise to an unseen generator? | **Yes, at the level of the supervised baseline.** TriCue-Mech reaches 0.776 on MuseTalk, level with B-AVH at 0.800 (CI contains 0); +0.075 above the same architecture trained on Wav2Lip and +0.056 above P2. The price is ~0.06 on Wav2Lip | Medium: same direction for 2 of 3 speakers, opposite for speaker 01 |
| Does adding real fakes to MechFake help? | **No** for the unseen generator (MuseTalk −0.024 ✓); it only helps the seen generator (Wav2Lip +0.016). Matches "1+1<2" (Cheng 2024) | Medium |
| What does each TriCue branch contribute? | The "AI-painted?" output ≈ r_art; m_sync adds only +0.014; m_lead ≈ 0. m_sync's value lies in **the "mismatch?" output** (sham 0.94; MuseTalk donor 0.83) | High (consistent across 3 seeds, mechanistically sound) |
| Is there a mechanism → generator map? | **No.** All 5 mechanisms are worse than every leave-one-out; no mechanism, when dropped, specifically hurts its matching generator | High for "no clear map"; low for the mechanism ranking (gaps ≤ 0.05, chosen on test) |
| Does the model recognise AI or processing? | **Processing.** Every artifact branch fails E6a and E6b; only B-AVH (AV-HuBERT grey-scale lip features) passes | High (FAR 0.79–1.00 vs 0.06–0.33 on real) |
| Does the model look at the painted part? | **Yes, with MechFake.** E7 on MuseTalk reaches 0.82 and 88% of clips, well above models trained on Wav2Lip (0.65–0.69 and 27–42%) | Medium to high |

### 16.2. Overall reading

1. **The main claim holds, but it is narrow.** Supported: training only on mechanism-based pseudo-fakes, with no generator, matches the supervised baseline on an unseen generator and localises the painted segment. **Not** supported: "better than the baseline", "a mechanism map", "recognises AI".
2. **The core weakness is the label definition, not the architecture.** MechFake teaches the model that a re-processed mouth region is AI, so the model also flags benign processing. §5.4 warned about this: E6a cannot be applied to MechFake because M1 and M2 are exactly the paste operation. To separate "AI" from "processing", **benign processing must be added as label 0** during training.
3. **The largest error in the pooled AUC is per-speaker calibration**, not separability: within a speaker, AUC is 0.95–0.99. This suggests **normalising scores against the same person's real videos** (reference clips, i.e. enrollment). It is cheap and matches the real use case: checking a video of a known person.
4. **TriCue should be slimmed down**: drop m_lead and keep two outputs ("AI-painted?" from r_art, "mismatch?" from m_sync). The name "three cues" no longer describes the model.
5. **B-AVH remains a hard baseline** and is the **only** model that passes E6. AV-HuBERT features (88 px grey-scale lips, learned from lip reading) are less sensitive to processing traces. Combining B-AVH, or its features, with a MechFake-trained r_art is a natural candidate.
6. **The 2×2 protocol with E6/E7 and per-speaker analysis is a valuable evaluation contribution.** It changes how the AUC numbers read: 0.78 pooled but 0.96 within a speaker, and part of the Wav2Lip model's 0.69 "recognises MuseTalk" is recognising the paste trace.

## 17. Limitations (to be stated with every number)

1. **Only 3 test speakers, 3 validation, 4 train** (part1 has only 10 speakers). No per-speaker CI; per-clip CIs are optimistic; one speaker (01) is enough to reverse the MechFake vs W2L conclusion.
2. Only two test generators (Wav2Lip, MuseTalk). Building a mechanism map from 2 generators is under-determined.
3. **The best configurations were selected on the test set.** `art_drop_m5` and `art_only_m4` score best, but were picked after seeing the test (out of 12 configurations), so they must not be reported as the proposed method. Validation sets currently differ between models, so there is no valid way to select yet.
4. Thresholds chosen on validation do not transfer to new speakers: FAR ranges 0.06–0.57, with seed-to-seed spread up to 0.23.
5. Random re-compression during training was not done; stage F learns on the branches' own training data (§15.7).
6. Mouth boxes are only 45–85 px, upscaled to 224 (v001's R1 question remains open).
7. MechFake is only of the "source" kind (original audio), except M5. The "mismatch" question is therefore learned only from shams and shifted reals.

## 18. Gate assessment and direction decision

| Gate | Criterion | Result | Passed? |
|---|---|---|---|
| G1 | Wav2Lip and MuseTalk > 0.65; MuseTalk > 0.64 | `art_all` 0.918 / 0.762 | ✔ |
| G2 | Some mechanism explains some generator | No clear map (§15.5) | ✘ |
| G3 | `tricue_mech` ≥ `tricue_w2l` on MuseTalk, without falling too far on Wav2Lip | MuseTalk +0.075; Wav2Lip −0.057 (still 0.933) | ✔ (2 of 3 speakers) |
| §7 success | Wav2Lip and MuseTalk ≥ 0.8, approaching or beating B-AVH | MuseTalk 0.776; level with B-AVH | ½ |
| §5.4 E6b | FAR ≈ real FAR | 0.54 vs 0.17 | ✘ |

**Recommendation: keep the MechFake direction, do not change direction, but do not go straight to part2.** Reasons:
- The positive evidence (G1, G3, E7, the "mismatch?" output) is enough to continue. The failures (G2, E6) point to **exactly what to fix** rather than refuting the direction.
- The two core flaws (processing treated as AI; per-speaker baseline shift) **can be fixed and tested on part1 right away** at low cost: CPU generation, ~1–2 h GPU of DINOv2 extraction, a few minutes of training per configuration. Going to part2 first would spend about 2 weeks of quota re-measuring a model already known to flag processing.
- Part2 (≥ 5 test speakers) is still **required** before any main conclusion, but should come after the method is locked in v002.1.

| Option | What | Cost | Assessment |
|---|---|---|---|
| **A. v002.1 on part1, then part2** (recommended) | Benign processing as label 0, re-compression, per-speaker normalisation, slimmer TriCue (§19 items 1–4), re-measure E6; then part2 | About 1 week (≈ 5–8 h GPU) + about 2 weeks of quota for part2 | Fixes the measured flaws and keeps part1 as a test bench |
| B. Part2 now | Collect 12–20 speakers, rerun v002 as is | About 2–3 weeks of quota | Per-speaker CIs, but carries the known E6 flaw |
| C. Change direction (e.g. stronger generators, face swap) | — | High | Wastes the evidence gathered; not recommended |

## 19. Follow-up research (by priority)

| # | Problem | Why (measurement) | Approach | Cost |
|---|---|---|---|---|
| 1 | **Benign processing as label 0 during training** | E6a 0.79–1.00; E6b 0.38–0.78 | Add to train/val, **as label 0**: real mouths pasted back unchanged (paste boundary only), and whole-frame processing (whole-frame downscale–upscale, sharpening, gamma, re-compression). Keep M2–M5 (regenerated content) as label 1. Question to measure: does the model still detect MuseTalk once it is no longer rewarded for spotting paste traces? Keep the test diagnostics unchanged for comparison | CPU generation; DINOv2 ~0.5 h; minutes of training |
| 2 | **Random re-compression during training** (CRF 18–35) for reals and MechFake | §4.1 not done; E6b | Augment at generation time, or add extra samples | As item 1 |
| 3 | **Per-speaker normalisation with reference clips** | Within a speaker 0.95–0.99, pooled only 0.78; speaker 01's reals score 0.73 | Subtract a baseline computed from a few known real clips of the same speaker (per-speaker z-score). Report both "no reference" and "with reference" modes | CPU, no retraining |
| 4 | **Slimmer TriCue**: drop m_lead; stage F learns on out-of-sample data (split train by speaker, or cross-fitting) | m_lead ≈ 0.5; the train-on-train flaw in §15.7 | 2 branches and 2 outputs; state clearly that each output answers one question | Minutes of training |
| 5 | **One shared validation set for configuration selection** | Validation differs between models; selecting on test is invalid | One fixed validation set for every model (real, sham and all MechFake mechanisms); select mechanisms and configurations before opening the test set | Free |
| 6 | **Combine B-AVH with a MechFake-trained r_art** | B-AVH is the only model passing E6; r_art localises better | Add a supervised AV-HuBERT branch trained on MechFake (no Wav2Lip), or late-fuse the two models' scores | Minutes of training |
| 7 | **Investigate speaker 01** | This speaker reverses the conclusion; real videos score high | Inspect per-cell scores on speaker 01's reals; compare sharpness, noise, lighting, the 1.8–2.0 Mbps bitrate; try the normalisation in item 3 | CPU |
| 8 | **Part2: 12–20 speakers, ≥ 5 test speakers** | No per-speaker CI yet | As planned in §5.1; generate real generators only for test (and Wav2Lip for the baselines) | About 2 weeks of quota |
| 9 | **A 3rd–4th test generator** (LatentSync, Diff2Lip, VideoReTalking…) | A mechanism map needs more generators | Test only; LatentSync first, since it also uses a VAE latent like MuseTalk | A few GPU hours per generator per part |
| 10 | **Mouth resolution (R1)** | Mouth boxes of 45–85 px upscaled to 224 | Compare `max_side` 640 vs 960/1280 for DINOv2 (AV-HuBERT uses the same boxes) | Re-extract DINOv2 ~1–2 h |
| 11 | Score and threshold calibration | FAR ranges 0.06–0.57 | Temperature scaling or isotonic regression on the shared validation set; keep AUC/AP as the main metrics | CPU |
| 12 | External test (AVLips, FakeAVCeleb, LipSyncTIMIT) | Cross-check on generators not used to design the mechanisms | After the method is locked | Depends on access |
| 13 | Voice cloning (6 cells) | Scope of v003 | Keep deferred | — |

**Proposed order:**
1. On part1 (v002.1): item 5 → items 1 + 2 → item 3 → item 4 (with item 6), then re-measure E6/E7 and the per-speaker results.
2. Part2 for the main conclusions: items 8 + 9.
3. Afterwards: items 10–12.

> **Closing note (14/10/2026).** The research was closed after v002: v001 and v002 together form one finished study. The follow-up work is being rebuilt as a separate project (v003: a Vietnamese lip-sync deepfake benchmark).
