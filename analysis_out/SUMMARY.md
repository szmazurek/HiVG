# Why distilled backbones "collapsed" under HiVG grounding

**Verdict.** The distilled checkpoints are healthy. The collapsed table rows come from our
port of HiVG's cross-modal bridge (`MACB`), and the failure is independent of which
backbone is loaded — OpenAI's own weights collapse identically.

But the sharper finding, which arrived last and reframes everything: **the bridge
contributes nothing to final accuracy even when it works.** Removing it entirely matches
the repaired model. So the bug was not "the bridge failed to help" — it was **active
injury from a module that is decoration**. Run on HiVG's own code path, the same ablation
returns +1.23 val / +0.45 test against their claimed +3.05 / +2.76 (§1b).

Checkpoint under test: `outputs/20033895_kd_vit_b16_to_b16_cc3m12m/checkpoints/last.ckpt`
(ImageNet zero-shot 74.18%, retrieval above its own DFN2B teacher).

*This document supersedes earlier drafts; several intermediate hypotheses recorded here
were refuted by later runs and are listed as such in §6.*

---

## 1. The ablation that reframes the result

RefCOCOg (`gref_umd`), stage 0, 60 epochs, identical recipe throughout — only the bridge
configuration differs. **All arms complete and evaluated.**

### 1a. Our port, distilled backbone

| configuration | flag | val | test |
|---|---|---|---|
| no bridge at all | `--macb_disable` | **74.51** @ep51 | **73.56** |
| bridge, all params, constant text | `--macb_text_const` | **74.14** @ep52 | **73.66** |
| bridge, correctly wired + normalised | `--macb_text_ln` | 73.82 @ep44 | 73.52 |
| bridge, correct layout, **un-normalised** | *(the published config)* | **26.45** | **25.58** |

Three healthy arms within **0.7 val and 0.15 test**. Test is dead even, and the arm fed a
learned *constant* in place of text is indistinguishable from the arm fed real captions.

### 1b. HiVG's own native path — same ablation, their code

`--disable_adapt_layer` on `model=ViT-B/16` with OpenAI weights, matched to their reference
run exactly (60 ep, lr 2.5e-4, batch 160, cosine, `clip_max_norm 0`):

| configuration | val | test |
|---|---|---|
| with MACB (`output_v100_oaiclip`) | 75.41 @ep47 | 75.30 |
| **without MACB** (`w7_native_no_macb`) | **74.18** @ep43 | **74.85** |
| **measured contribution** | **+1.23** | **+0.45** |
| *claimed in their Table (MACB ✗→✓)* | *+3.05* | *+2.76* |

Their gain does not reproduce at the size claimed on their own code: **~40% of it on val,
~16% on test.** Caveat on precision — their MACB-on row reads 76.53 and we reproduce that
configuration at 75.41, so there is already a ~1.1-point offset between this environment
and their table. That bounds the resolution here to roughly a point; it does not rescue a
3-point claim.

Three things follow.

**The bridge earns nothing (ours) to ~1 point (theirs).** Its main real effect is faster
warmup. ⚠ **Never read this ablation before ~ep25** — the early gap favouring the bridge is
a warmup transient that closes by ep20 and reverses; it misled me once (§6).

**The residual is not cross-modal.** `--macb_text_const` keeps every parameter and deletes
only the language, and it matches the no-bridge arm. Whatever the bridge does, it is not
conditioning vision on the caption.

**The broken bridge was actively destructive.** 26.45 is **48 points below simply not
having a bridge**. `--macb_text_ln` does not unlock anything; it restores the no-bridge
baseline.

---

## 2. Repaired results vs the published table

All stage-0, 60 epochs, `--distilled_text_tower --macb_zero_init --macb_text_ln`,
`--clip_max_norm 1.0`.

| dataset | split | published | **repaired** | OpenAI | HiVG paper |
|---|---|---|---|---|---|
| RefCOCOg | val | 23.9 | **73.82** | 75.12 | 78.29 |
| RefCOCOg | test | 23.8 | **73.52** | 75.3 | 78.79 |
| RefCOCO+ | val | 29.9 | **72.97** | 76.1 | 78.06 |
| RefCOCO+ | testA | 35.8 | **78.06** | 83.2 | 83.81 |
| RefCOCO+ | testB | 23.58 | **64.27** | 64.7 | 68.11 |
| ReferIt | test | 37.7 | **68.54** | 69.2 | 71.44 |
| ReferIt | val | — | **71.34** | — | — |
| Flickr | val | 65.4 | **77.29** ⚠ | 81.5 | — |
| Flickr | test | — | **78.81** ⚠ | — | — |

⚠ Flickr is 19/60 epochs only — training was stopped early to free GPUs, and this is the
epoch-16 checkpoint. Report as partial-budget; it is already +13 over the published
number and would likely go higher.

RefCOCO+ testB is level with OpenAI (64.27 vs 64.7) and ReferIt within 0.7.

The `OpenAI` column above is HiVG's **native** path. Note it is not the same thing as
OpenAI weights through *our* port — see §2a, which is the comparison the repaired distilled
rows should actually be read against.

### 2a. A residual port deficit that is NOT the bridge

Budget-matched, both bugs fixed, OpenAI's own weights, RefCOCOg, 60 ep, `clip_max_norm 0`
(`w5_oai_fixed60`; killed by the time limit at ep57/60, evaluated from its ep-35 best —
the cosine tail had been declining since ep35, so no best was lost):

| path, identical OpenAI weights | val | test |
|---|---|---|
| HiVG native | 75.41 | 75.30 |
| our port, both bugs fixed | **72.75** @ep35 | **72.49** |
| deficit | **−2.66** | **−2.81** |

This is a **null result for the repairs against the paper's baseline** — expected, since §1
says the bridge cannot matter either way. But it isolates something new: our port sits
~2.7 points below native *on the same weights, with the bridge ablated to irrelevance on
both sides*, so MACB cannot explain it.

It also reframes §2. Compared against the *port's own* OpenAI number rather than the native
one, the repaired distilled backbone is **ahead**: 73.82/73.52 vs 72.75/72.49, i.e. **+1.07
val / +1.03 test**. So the distilled checkpoint is not what limits §2 — most of the
remaining gap to the paper's 78.29 is this port deficit plus HiLoRA (§6). ⚠ Not perfectly
matched: the distilled run used `--clip_max_norm 1.0`, this one 0 (to match native). A
`clip_max_norm 1.0` OpenAI arm would close that hole and is the cheapest next run.

Candidates, none tested: native `:804` *wraps* OpenAI's vision module in place while the
port *replaces* the tower (different LoRA/param scope); `--distilled_text_tower` swaps HF
`CLIPTextModel` for open_clip's text tower (verified numerically equivalent in isolation,
but it changes what `hidden_states` MACB and CLC consume); `ml_visual_ln`; `logit_scale`
left at OpenAI's 100 vs the checkpoint's own value.

---

## 3. Why severity looked backbone-shaped

Severity is ordered by `len(extract_text_layer)` (`HiVG.py:727-738`), not by anything about
the checkpoint. `K` is the dose:

| dataset | `extract_text_layer` | published | OAI | gap |
|---|---|---|---|---|
| RefCOCO | `[12]` (K=1) | 82.11 | 85.2 | −3.1 |
| Flickr | `[12]` (K=1) | 65.4 | 81.5 | −16.1 |
| ReferIt | `[6,12]` (K=2) | 37.7 | 69.2 | −31.5 |
| RefCOCO+ | `[6,12]` (K=2) | 29.9 | 76.1 | −46.2 |
| RefCOCOg | `[1..12]` (K=12) | 23.9 | 75.12 | −51.2 |

The two datasets that survived are exactly the two that never exercise the multi-layer
branch. Dose-response on RefCOCOg at fixed everything-else (20 ep, distilled):

| K | no `text_ln` | with `text_ln` |
|---|---|---|
| 1 | 67.52 (28 ep) | 71.28 / 71.63 |
| 2 | 26.82 / 26.08 | — |
| 12 | 26.45 / 25.58 | 72.39 / 71.65 |

**K=1 was never healthy either** — it stalls ~10 epochs before escaping. `text_ln` removes
the stall (ep1: 11.3 → 40.1), so the K-cliff is severity of one pathology, not two
mechanisms.

---

## 4. Mechanism: two bugs that mask each other

### Bug 1 — un-normalised multi-depth concat (`HiVG.py:443-451`, our `macb.py`)

Text hidden states from different depths are concatenated raw. Their norms differ ~4x
(distilled 9.45 at layer 1 → 35.07 at layer 12; OpenAI 8.5 → 14.66) and CLIP's space is
strongly anisotropic, so the input to `gate` is **effectively rank 1** (condition number up
to 9.98e11). `gate` emits near-degenerate keys, cross-attention cannot discriminate text
tokens, and the bridge compensates with **magnitude** — ending at **46x the visual features
it is added to**, swamping the frozen backbone.

Trained-bridge measurements, perfect separation:

| arm | eff. rank (keys) | delta/visual | accuracy |
|---|---|---|---|
| `[1..12]` + `text_ln` | 3.86 | 0.110 | 72.4 |
| `[12]` plain (K=1) | 2.13 | 0.133 | 67.5 |
| `[6,12]` plain (K=2) | 1.05 | 1.066 | 26.8 |
| `[1..12]` plain (K=12) | 1.17 | **45.95** | 22.3 |

### Bug 2 — transposed cross-attention input (`HiVG.py:452`)

```python
adpt_text_states = adpt_text_states.permute(1, 0, 2)   # B L H --> L B H
```

fed to `CLIP_Cross_Attention`, whose docstring says `Batch x Time x Channel`. Inside,
`_shape(k_proj(text), -1, bsz)` does `.view(bsz, -1, heads, head_dim)`. It does not crash
because `k_proj` emits a fresh **contiguous** tensor (removing the usual non-contiguous
`.view()` guardrail), `seq_len=-1` infers back to exactly 77, and the shape assertion at
`:181` passes. Shapes right, contents wrong.

Verified element-for-element: key slot `s` of batch slot `b` reads
`(token, sample) = divmod(b*L + s, B)`. Severity scales with batch size:

| B | own-caption keys | padding keys (past EOS) |
|---|---|---|
| 16 | 6.49% | 85.55% |
| 77 | 1.30% | 85.73% |
| **160 (HiVG's)** | **0.65%** | **85.55%** |

The own-caption rate is exactly `1/B` — chance. Captions run 4-24 tokens of 77 slots, so
most keys are padding. Measured batch leakage: native **0.159919**, our port **0.000000**.

### How they interact

Bug 2 **masks** bug 1. With scrambled input there is no loss-reducing direction involving
text, so `gate` settles into a small content-free offset and a rank-1 bottleneck costs
nothing. Our port fixed bug 2 structurally, which made the bridge *live* — and a live
bridge with a rank-1 gate is catastrophic.

**Confirmed by experiment, not argument.** Reproducing bug 2 in our port
(`--macb_hivg_layout`, no `text_ln`), RefCOCOg K=12, 20 epochs:

| backbone | layout | `text_ln` | val | test |
|---|---|---|---|---|
| OpenAI | correct | ✗ | **29.70** | — |
| OpenAI | **HiVG's** | ✗ | **71.73** | **72.02** |
| OpenAI | correct | ✓ | 72.65 | 72.81 |
| distilled | correct | ✗ | 26.45 | 25.58 |
| distilled | **HiVG's** | ✗ | **69.79** | **70.47** |
| OpenAI | HiVG native path | ✗ | 75.41 (60 ep) | 75.3 |

Reproducing the bug rescues the un-normalised bridge from 29.70 to 71.73 with no
normalisation at all, on both backbones.

---

## 5. HiVG's own code: the defect is there, and their ablation misattributes

Audited on their trained native checkpoint (`output_v100_oaiclip/gref_umd/20260723_105828`,
`model=ViT-B/16`, K=12, val 75.41). Scripts: `analysis/native_bridge_audit.py`,
`analysis/native_bridge_content.py`.

**The rank-1 defect is present in the original**, so `--macb_text_ln` is an *addition* to
the paper recipe, not a restoration of it:

| slot | exec | cond(concat) | eff. rank | delta/visual |
|---|---|---|---|---|
| 0 | 1 | 4.97e7 | 1.02 | 0.208 |
| 1 | 4 | 5.85e7 | 1.02 | 0.413 |
| 2 | 8 | 9.17e7 | 1.01 | 0.793 |
| 3 | 12 | 6.57e7 | 1.02 | 0.501 |

**Their bridge is not caption-conditioned.** Relative change in sample 0's bridge output
when a caption is replaced — same trained weights, only `:452` toggled:

| layout | s_own | s_other |
|---|---|---|
| HiVG as shipped | 0.016 – 0.031 | 0.002 – **0.027** |
| layout-correct | 0.032 – 0.151 | **0.00000** |

As shipped it reacts to an *unrelated* caption almost as strongly as its own (slot 3: 0.027
vs 0.031). Caveat: the layout-correct row uses weights trained under scrambling, so its
absolute `s_own` understates a properly trained model; the meaningful contrasts are
`s_other` (0.027 → exactly 0) and the own/other ratio.

**Why their ablation reports +3.** MACB is **49,115,136 trainable params = 21.2% of the
model**, bolted onto a frozen backbone:

| component | params | share |
|---|---|---|
| `cross_mlp` | 18,889,728 | 38.5% |
| `cross_gate` | 18,877,440 | 38.4% |
| `cross_attn` | 9,449,472 | 19.2% |
| `cross_adaptive_weights` | 1,892,352 | 3.9% |

Their negative row removes the whole module, confounding capacity with cross-modality —
and a module fed 0.65%-own-caption noise cannot earn 3 points by conditioning vision on
language. Note their `--enable_adaptive_weights`-off path is not the ablation either: it
feeds raw 512-d text into a `Linear(768,768)` and only runs under `mixup_pretrain`, so
their ✗ row must be an empty `adapt_layer`.

Our §1 ablation supplies the two rows their table cannot: capacity held fixed with language
removed (`--macb_text_const`), which tracks the no-bridge arm; and their own ✗ row rerun on
their own code (§1b), which returns +1.23/+0.45 rather than +3.05/+2.76.

**What is left of MACB's +3, then?** Not conditioning — the module receives no usable
language (above). Not capacity either, in our port, since `--macb_text_const` keeps all
49.1M params and gains nothing. On their path a ~1.2-point val effect does survive, and the
honest description of it is *an unexplained residual of roughly a point*, not a cross-modal
mechanism. The most likely candidates are the extra trainable capacity interacting with
their frozen backbone and the faster warmup buying a better spot on a 60-epoch cosine —
both testable, neither tested.

---

## 6. Refuted along the way (do not re-run)

Recorded because several were promoted to "the cause" prematurely.

* **Distilled backbones lack spatial features.** MaskCLIP dense probe: student 57.5 /
  teacher 53.5 / OpenAI 54.0 (chance 20.9). Refuted.
* **Gradient clipping was the blocker.** Real defect (`--clip_max_norm` defaults to 0,
  `engine.py:62`) but gref_umd shows no loss spikes. Not the story.
* **`ml_visual_ln` scale mismatch**, **`macb_zero_init`** — neither fixes anything.
* **Attention saturation at init.** Looked decisive on one seed; over 5 seeds all K are
  equally saturated (K=1 entropy 2.505 ± 0.534). Does not discriminate.
* **Gradient starvation.** `cross_mlp[-1]` is zero-init in every arm and grows to 4.5-10.4
  in all of them. The failing arms learn to inject *more*, not less.
* **"The network learns to ignore the bridge."** Wrong — native `delta/visual` is 0.21-0.79,
  *larger* than our fixed port's 0.110. The bridge is not ignored, it is content-free.
* **"The bridge earns its keep in our fixed code."** Claimed from epochs 0-6 of the
  ablation, where no-bridge trails by ~4 points. Wrong: that is a warmup transient, the
  gap closes by ep20 and reverses. Never read this ablation before ~ep25.
* **HiLoRA stages help.** They do not, here. Stage 1 opens at 72.1 (from 73.82) and peaks
  at 72.86; stage 2 peaks at 73.59. Neither beats stage 0.
* **`ml_text_feat_perceiver`** (`HiVG.py:883`) is dead code — defined, never used in
  `forward`.

### The HiLoRA "identical eval" puzzle — resolved, not a bug

Stage-1 eval reproduced stage-0's numbers to 2 d.p. on all five splits, and every `lora_B`
in the stage-1 checkpoint is exactly 0.0. Cause: `hivg_train.py:355-366` validates the
loaded model on entering a stage, seeds `best_accu` with it, and saves it as `epoch: -1`
(their comment: `# Prevent negative optimization`). No later epoch beat the seed, so
`best_checkpoint.pth` correctly stayed the stage-0 snapshot — taken before any optimizer
step, hence `lora_B == 0`. Verified: 0 of 724 shared tensors differ between the stage-0 and
stage-1 checkpoints. LoRA is trainable, is in the optimizer (+2,555,904 params at stage 1),
and does receive gradient (`analysis/lora_grad_probe.py`, max |grad| 8.09e-2).

---

## 7. Four genuine but minor defects found and fixed (all default-off flags)

1. **Text-tower mismatch.** The port transplanted only the distilled *vision* tower and
   kept OpenAI's text tower — unrelated 512-d spaces. ImageNet zero-shot 0.064% (below
   chance) vs 74.258% matched. Fixed by `--distilled_text_tower`.
2. **Gradient clipping disabled** by default. `--clip_max_norm 1.0`.
3. **Bridge init inflating the residual 20x.** `--macb_zero_init`.
4. **Silent QuickGELU mismatch.** open_clip only *warns*; `--standardvit_arch
   ViT-B-16-quickgelu` for OpenAI/DFN2B weights.

Only #1 is load-bearing for a results row: it, not `text_ln`, is the likely driver of
Flickr's recovery (K=1, where the concat has nothing to imbalance). Untested separately —
a Flickr arm with `text_ln` off would settle it.

---

## 8. Scope: what else these checkpoints touch

`MACB` is imported only by `HiVG/models/{standardvit,loopvit}_bridge.py` and
`src/downstream/models/hivg_loopvit.py` — all grounding. **DIET (unlearning) and CLIP-LoRA
(few-shot) are unaffected.**

Every native `src/downstream` config uses `extract_text_layers: [12]` (K=1), which is why
the bViT / b²ViT rows never collapsed. But K=1 costs ~10 epochs of stall, so those rows may
be understated; `macb_text_ln: true` is now exposed as a config key (default `False`) to
test that.

---

## 9. Reproduction

```bash
# Phase 0 gate: 0.064% (distilled vision + OpenAI text) vs 74.258% (matched)
python scripts/preflight_hivg_text_tower.py

# the defect, in HiVG's own trained model
python analysis/native_bridge_audit.py   HiVG_outputs/output_v100_oaiclip/gref_umd/20260723_105828/best_checkpoint.pth
python analysis/native_bridge_content.py HiVG_outputs/output_v100_oaiclip/gref_umd/20260723_105828/best_checkpoint.pth

# the ablation (§1a) -- our port, 60 epochs, gref_umd
bash train_and_eval_script/chain9.sh     # --macb_disable, then --macb_text_const
# the same ablation on HiVG's OWN path (§1b) -- --disable_adapt_layer, ViT-B/16 + OpenAI
bash train_and_eval_script/chain11.sh
# budget-matched OpenAI through the port, both bugs fixed (§2a)
bash train_and_eval_script/chain8.sh
# the layout mechanism (§4)
bash train_and_eval_script/chain7.sh     # --macb_hivg_layout, no text_ln

# eval-only recovery when the time limit kills a run after its last best checkpoint
bash train_and_eval_script/eval_w5.sh
```

Launch traps, both hit in practice:
* `pgrep -f "hivg_train.py"` **self-matches the calling shell** — wait loops built on it
  hang forever. Gate on an explicit PID, and verify it with `kill -0` (a `pgrep` for the
  chain script itself also returns transient matches).
* Killing a chain shell does **not** kill its `phase3_stage0.sh` children; a relaunch then
  collides on `--master_port` and overwrites logs. Kill children explicitly, then the CUDA
  workers from `nvidia-smi --query-compute-apps=pid`, and confirm 0 MiB.

---

## 10. Open items

Closed since the last revision: chain9 arm B (§1a), chain8 (§2a), chain11 (§1b) — all three
complete and evaluated. `eval_w5.sh` is the eval-only recovery pattern for a run killed
after its last best checkpoint but before its eval stage; reuse it, don't retrain.

Ranked by information per GPU-hour:

1. **The §2a port deficit** — ~2.7 points on identical OpenAI weights, not attributable to
   MACB. This is now the largest unexplained quantity in the writeup and it gates how the
   distilled rows should be read. Cheapest probe: OpenAI through the port with
   `--clip_max_norm 1.0` (matches the distilled arms) and with `--distilled_text_tower`
   off, to split the tower-swap from the rest.
2. **RefCOCO (`unc`) has no corrected run at all** — the one table row with no repaired
   number. 60 ep, 4 GPUs.
3. **Flickr at full budget** — current number is 19/60 epochs and already +13 over
   published.
4. **bViT / b²ViT rows** via the native `src/downstream` path with `macb_text_ln: true`
   (§8) — those rows never collapsed (K=1) but may be understated by the ~10-epoch stall.
5. **Eval-batch-size invariance on native HiVG** (~10 min, never run despite being flagged
   repeatedly). If native eval accuracy moves between `--batch_size 8` and `160`, the
   scrambled bridge does carry signal and §5's content-free claim needs weakening.
6. **Flickr `text_ln`-off control** — separates defect #1 from the normalisation fix.
7. **HiLoRA schedule tuning** (lower stage-1 lr, or a shorter stage 0) — the only untried
   route to the paper's 78.29, given §6 says the stock curriculum slightly hurts.
8. Whether a 49M-param adapter with no cross-modal role is worth keeping at all (§1
   suggests not; §1b's residual ~1.2 val on the native path is the case for keeping it).
