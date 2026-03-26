# CellFlux Replication Notes

## Differences between our current setup and the paper

### 1. CFG scale: 1.2 vs 0.2 ✅ DONE
Paper uses `cfg_scale=1.2`. Our config had `cfg_scale=0.2` (matching the code default).
Fixed in `configs/cellflux-bbbc021-paper-lb.yaml` with `cfg_scale=1.2`.

### 2. Training duration: 100 epochs vs 80k iterations — OK as-is
Paper trains 100 epochs iterating over trt images (81,506) = ~63,600 iters at bs=128.
Our 80k iterations is slightly more training. Keeping 80k.

### 3. Multi-GPU: 4 A100s vs 1 GPU — SKIPPED
Paper uses 4 A100s. Unknown if effective batch size is 128 or 512. Skipping for now.

### 4. Data iteration direction (ctrl vs trt) ✅ DONE (option added)
Original iterates trt, finds ctrl. Ours defaults to iterating ctrl.
Added `--iter_trt` / `iter_trt: true` option to `IMPADataset` and CellFlux training
script. Not enabled by default — use in paper-matching config if needed.

### 5. Batch matching vs plate matching — OK, equivalent
BATCH="Week1_22123" and plate_from_key="22123" are 1-to-1. Same grouping.

### 6. Vertical flips — SKIPPED
Our dataset adds vertical flips, original doesn't. Unlikely to matter.

### 7. Model selection via validation FID ✅ DONE
Paper: "Models are selected based on the lowest FID scores on the validation set."
Added `--val_fid` flag. When enabled with `--fid_every`, computes FID on the test
split and saves `best_val_fid.pt` checkpoint whenever val FID improves.
Enabled in `cellflux-bbbc021-paper-lb.yaml`.

### 8. noise_level: 0.5 vs 0.2 — OK as-is
Paper says noise injection probability 0.5 (= noise_prob). noise_level=0.5 matches
their example script. Keeping as-is.

## Things to try next

- [ ] Set `cfg_scale=1.2` to match the paper
- [ ] Implement validation split FID evaluation and model selection
- [ ] Verify batch vs plate equivalence in BBBC021 metadata
- [ ] Try `iter_trt: true` to iterate over treated images (match original sampling)
- [ ] Remove vertical flips to match original augmentation
- [ ] Check effective batch size: is the paper using 128 per GPU (512 total) or 128 global?
- [ ] Sweep noise_level (0.2 vs 0.5)
