# SOLUTION.md

## Reproducibility Instructions

**Platform:** Kaggle Notebook, GPU T4  
**Python:** 3.12, PyTorch 2.x, torchvision, scikit-learn

**Dataset:** Add `fedesoriano/cifar100` from Kaggle Datasets.  
Files available at `/kaggle/input/datasets/fedesoriano/cifar100/`.

**Steps to reproduce:**

1. Clone repository:git clone <your-fork-url>
cd SMILES-2026-ZO-Limited-Resnet
pip install -r requirements.txt

2. Run feature extraction and pseudo-inverse head initialization (notebook cell):
```python
# Extract ResNet18 backbone features for all 50k train images
# Solve W = lstsq(X, Y_onehot) — closed-form, no gradients
# Save result to /kaggle/working/pinv_head.pt
```

3. Run evaluation:
python validate.py --data_dir ./data --batch_size 32 --n_batches 256 --output results.json

---

## Final Solution Description

### Core idea: Pseudo-inverse head initialization

The main insight is that the ZO budget (8192 samples, 256 steps) is far too
small to learn a 100-class linear head from random initialization using
noisy SPSA gradient estimates.

Instead, we initialize the head analytically using the **closed-form
least-squares solution**:
W = pinv(X_train) @ Y_onehot

where `X_train` is a (50000, 512) matrix of ResNet18 backbone features,
and `Y_onehot` is the (50000, 100) one-hot label matrix.  
Solved via `torch.linalg.lstsq` — a single SVD decomposition, no iterations,
no gradients, no optimization.

This is implemented in `head_init.py` which is a student-editable file.
The backbone is never modified — only the final `fc` layer weights are set.

### ZO fine-tuning on top

After pseudo-inverse initialization (59.76%), we apply SPSA zero-order
optimization with:
- `lr = 1e-6` (very small to avoid destroying the good initialization)
- `eps = 1e-3` (perturbation magnitude)
- Rademacher ±1 perturbation vectors
- **Greedy rejection**: if a step increases loss, weights are rolled back

This gives a small additional improvement: 59.76% → 59.81%.

### Modified files

| File | Change |
|---|---|
| `head_init.py` | Loads pseudo-inverse weights from `pinv_head.pt` |
| `zo_optimizer.py` | SPSA with greedy rollback, lr=1e-6 |
| `augmentation.py` | Added RandomCrop, ColorJitter, RandomErasing |
| `train_data.py` | Custom Dataset reading CIFAR-100 pickle directly |

---

## Experiments and Failed Attempts

### Adam + SPSA, lr=0.05
Loss diverged: 5.4 → 15.3 in 30 steps.
Adam amplifies noisy SPSA gradients at this scale.

### SGD + SPSA, lr=1e-4, random init
Loss slowly decreased but accuracy stayed at 1.3% after full budget.
ZO alone cannot learn from scratch in 256 steps on CIFAR-100.

### SGD + SPSA, lr=1e-4, after pinv init
ZO destroyed the good initialization: 59.76% → 23.91%.
lr too large for already well-tuned weights.

### Logistic Regression via sklearn
Achieved 64.62% but uses gradient-based optimization internally —
methodologically inconsistent with the zero-order spirit of the task.
Replaced with pseudo-inverse (closed-form, no gradients).

### SPSA without greedy rollback, lr=1e-6
Loss flat (~4.41 ± 0.01), marginal degradation due to noise.
Adding rollback stabilized results: 59.76% → 59.81%.
