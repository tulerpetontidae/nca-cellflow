# Memo

## Digital petri dish — IMPLEMENTED

Pool-based NCA training where cells persist and evolve across iterations.

**Script:** `scripts/train_petridish.py` (wandb project: `nca-petridish`)
**Config:** `configs/petridish-base.yaml`
**Key components:** `src/nca_cellflow/pool.py` (ReplayPool), `LabeledImageBank` in dataset.py

**Architecture:** LatentNCA with dose conditioning: `[embed(compound) || dose_proj(log10(dose)) || z]` → FiLM. 60 NCA steps, step_size=0.005, tanh.

**Training mechanics:**
- 2048-slot GPU replay pool, states recycled when iter > 4
- 4 transition types: DMSO→DMSO (homeostasis), DMSO→drug, drug→drug (hold), drug→DMSO (recovery)
- Homeostasis perturbation: 3 wrong-label NCA steps before recovery
- Z drift: Brownian motion `z_new = sqrt(1-β)*z_old + sqrt(β)*N(0,1)`
- Checkpoint sampling: D evaluates random step from [T-10, T]
- Intermediate regularization: null-class loss keeps cells on manifold
- Style reconstruction: forces z usage

**Data notes:** BBBC021 has no temporal data (Weeks = batch, not exposure time). 84 unique (compound, dose) pairs, all ≥252 images. Dose is co-embedded with compound, not discretized.

## Large-image demo

Run the NCA on images much larger than 48x48 training crops. The architecture is fully convolutional — every operation (Sobel, pointwise MLP, FiLM) is resolution-independent, works at any size with zero code changes.

**Demo concept:** Large FOV of control cells running continuously (ctrl→ctrl dynamics — cells look alive). User drops a drug → swap FiLM vector → watch morphological transformation propagate across the screen. Can remove drug, try combinations, do sequential treatments.

**Considerations:** May want more NCA steps for larger images. Normalization must match training. Original BBBC021 has 96x96 crops and larger uncropped fields of view.

## Predicting unseen perturbations (forward)

Current model uses fingerprint conditioning (`nn.Linear(1024, 32)`), so in principle any new compound with a known structure can be fed through without retraining.

**Limitation:** 34 training compounds is very few to learn a general structure→morphology map. The linear projection likely memorizes rather than interpolates smoothly.

**To test:** Leave-one-out evaluation — retrain with 33 compounds, predict the held-out one. Compare to real with FID. Especially interesting within MoA families (hold out docetaxel, see if taxol training transfers).

**To improve:** Better molecular representations (MolBERT, ChemBERTa, CDDD) instead of Morgan FP. Train on larger compound libraries (JUMP-CP, RxRx have thousands). Replace linear projection with small MLP. Regularize embedding smoothness.

## Inverse problem: unknown drug → embedding

Given observed (ctrl, treated) image pairs from an unknown drug, optimize the embedding vector to make the NCA reproduce the observed effect. Fully differentiable — works out of the box.

**Approach:** `embed* = argmin_e E_z[||NCA(ctrl, [e, z]) - observed_trt||²]` — optimize 32-dim embedding, average over z samples to marginalize stochasticity. ~500 gradient steps.

**What it gives you:** Position in the learned perturbation space → nearest-neighbor MoA prediction. More powerful than fingerprint similarity because it captures *functional* similarity (structurally different drugs with similar morphological effects map nearby).

**Richer than standard morphological profiling:** Instead of extracting features and clustering (Cell Painting approach), you invert through a causal generative model. The embedding space is structured by how perturbations mechanistically affect morphology. Plus you can inspect NCA dynamics (dx maps, trajectory) to understand *why* compounds are similar.

**Amortized variant:** Adapt `NCAStyleEncoder` (predicts `[embed || z]` from generated image) to take (ctrl, trt) input — faster than per-sample optimization.

## z states vs dosage

Investigate correlation between NCA hidden/latent states and drug dosage effects. Do hidden channels encode dose in a structured way? Could z states form a gradient as dosage increases?

## Connection to ABM / Cramer et al.

The NCA is conceptually a learned agent-based model — local rules producing emergent dynamics, just like PhysiCell. Strongest angle: "we replace hand-crafted ABM rules with a learned local update rule." Could train NCA on PhysiCell simulation output (cell grids, 1 pixel = 1 cell) as a proof of concept. The architecture barely needs to change — it's mostly a data representation swap.
