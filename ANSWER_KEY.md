# Answer Key: Expected Audit Findings

This document catalogs all intentionally planted issues in this benchmark repository for verifying ReproAudit accuracy.

## Expected Static Metric Statuses

| Metric | Expected Value | Expected Status | Rationale |
|--------|---------------|-----------------|-----------|
| has_readme | true | good | README.md exists |
| has_license | true | good | MIT LICENSE exists |
| has_lock_file | false | **concerning** | No uv.lock, poetry.lock, etc. |
| python_version_specified | true | good | `requires-python` in pyproject.toml |
| unpinned_deps_count | ~14 | **critical** | 7 in pyproject.toml + 7 in requirements.txt without version specifiers |
| vulnerability_count | >=1 | needs_improvement | pillow==9.5.0 has known CVEs |
| lint_error_count | 0-2 | good / needs_improvement | Code is syntactically valid |
| lint_warning_count | ~10-20 | good | Unused imports, etc. |
| format_error_count | >=1 | needs_improvement | data_loader.py has bad formatting |
| type_error_count | >=1 | needs_improvement | preprocess.py compute_heart_rate returns str for float |
| avg_complexity | 5-8 | needs_improvement | train.py has high CC, others are simple |
| maintainability_index | ~50-65 | needs_improvement | Mixed quality across files |
| dead_code_count | ~5-7 | **concerning** | 3 in evaluate.py + 2 in utils.py |
| type_hint_coverage_pct | ~40-50% | **concerning** | model.py/preprocess.py good, train.py/data_loader.py/utils.py poor |
| docstring_coverage_pct | ~60-70% | needs_improvement | Most public functions documented |
| duplicate_pct | ~3-7% | needs_improvement | Duplicate eval logic in evaluate.py |
| unseeded_random_count | 2 | **concerning** | train.py (imports torch+numpy, no seeds) + data_loader.py (imports numpy, no seed) |
| hardcoded_path_count | 2 | needs_improvement | Two `/home/researcher/...` paths in train.py |
| secret_pattern_count | >=3 | **critical** | API_KEY and WANDB_API_KEY in utils.py match multiple patterns |
| test_file_count | 2 | **concerning** | Only test_model.py and test_preprocess.py |
| has_ci_config | false | needs_improvement | No .github/workflows or similar |
| has_containerization | false | needs_improvement | No Dockerfile, docker-compose, etc. |
| notebook_count | 3 | neutral | Informational only |
| notebooks_with_outputs_pct | 33% | **concerning** | 1/3 notebooks (01_data_exploration) has outputs |
| notebook_execution_issues | ~3 | **concerning** | Out-of-order exec (3<5) + missing counts (6,7) in notebook 01 |

## Expected Agent-Discovered Issues

### Documentation Quality

| Severity | Title | File | Description |
|----------|-------|------|-------------|
| warning | Incomplete README missing setup instructions | README.md | No installation, usage, or environment setup instructions. No description of how to run training or reproduce results. |
| positive | Well-documented model with type hints | src/model.py | All classes and methods have complete docstrings and type annotations. |

### Reproducibility Assessment

| Severity | Title | File | Description |
|----------|-------|------|-------------|
| blocker | External dataset with no download link or instructions | README.md, src/train.py | MIT-BIH Arrhythmia Database mentioned by name but no URL, DOI, or download script provided. Data dir is a hardcoded local path. |
| blocker | Hardcoded absolute paths prevent reproduction | src/train.py:16,103 | `"/home/researcher/data/mitbih"` and `"/home/researcher/models/best_model.pth"` are machine-specific paths. |
| warning | No random seed configuration | src/train.py | Imports torch and numpy but never calls torch.manual_seed() or np.random.seed(). Training results are non-deterministic. |
| warning | DataLoader workers without seed propagation | src/train.py:37-41 | `num_workers=4` but no `worker_init_fn` to propagate seeds to worker processes. |
| warning | Unseeded numpy random in data augmentation | src/data_loader.py:27 | `np.random.normal()` called during training data loading without any seed. |
| warning | No lock file for dependencies | pyproject.toml | requirements.txt and pyproject.toml present but no lock file (uv.lock, poetry.lock, etc.) to pin transitive dependencies. |
| warning | Known vulnerability in pinned dependency | requirements.txt:9 | pillow==9.5.0 has known security vulnerabilities (CVEs). |
| warning | Committed notebook outputs | notebooks/01_data_exploration.ipynb | Cell outputs committed to version control may not match re-execution. |
| warning | Out-of-order notebook execution | notebooks/01_data_exploration.ipynb | Execution counts [1,2,5,3,4,8] indicate cells were run out of order, suggesting hidden state dependencies. |
| warning | Non-deterministic output filenames | notebooks/03_visualization.ipynb (cell 4) | `datetime.datetime.now().strftime(...)` produces different filenames on each run. |
| warning | Non-deterministic groupby iteration | notebooks/03_visualization.ipynb (cell 3) | `df.groupby('class').mean()` without `sort=True` has non-deterministic row ordering. |
| recommendation | Consider uv for dependency management | -- | No lock file exists; uv can generate a deterministic lock file. |
| recommendation | Consider marimo for reproducible notebooks | -- | .ipynb files present; marimo eliminates hidden state. |

### Code Quality Analysis

| Severity | Title | File | Description |
|----------|-------|------|-------------|
| warning | High cyclomatic complexity in training function | src/train.py:13 | `train_model()` has deeply nested if/elif/else logging logic with CC > 10. |
| warning | Dead code functions | src/evaluate.py, src/utils.py | `plot_confusion_matrix`, `_old_evaluate`, `_compute_per_class_metrics` (evaluate.py) and `_deprecated_data_split`, `_unused_helper` (utils.py) are never called. |
| warning | Duplicate evaluation logic | src/evaluate.py | `compute_accuracy()` and `compute_f1()` duplicate functionality already in `evaluate_model()`. |
| warning | Type error in heart rate computation | src/preprocess.py:92 | `compute_heart_rate()` returns `"insufficient peaks"` (str) but declares return type `float`. |
| warning | Poor code formatting | src/data_loader.py | Missing spaces after colons, commas, and around assignment operators throughout the file. |

### Methodology Alignment (with manuscript)

| Severity | Title | File | Description |
|----------|-------|------|-------------|
| warning | Learning rate mismatch | src/train.py:22 | Paper states lr=0.001 (Section 2.4), code uses lr=0.01 -- a 10x discrepancy. |
| warning | No cross-validation implementation | src/train.py | Paper claims 5-fold cross-validation (Section 2.4), but code performs a simple train/val split. |
| warning | Missing data augmentation techniques | src/data_loader.py | Paper describes three augmentation methods (signal warping, noise injection, time-shift), but code only implements Gaussian noise. Signal warping and time-shift are absent. |
| warning | GPU requirements undocumented in code | -- | Paper specifies NVIDIA V100 with 16GB (Section 2.5), but code has no hardware documentation or GPU memory checks. |

## Expected Category Assessments

| Category | Assessment | Rationale |
|----------|-----------|-----------|
| Documentation Quality | **needs_improvement** | 1 warning (incomplete README) |
| Reproducibility Assessment | **critical** | 2 blockers (no data source, hardcoded paths) |
| Code Quality Analysis | **concerning** | 5+ warnings |
| Methodology Alignment | **concerning** | 4 warnings |
| **Overall** | **critical** | Any category critical -> overall critical |

## How to Use This Benchmark

### Without manuscript (skip methodology alignment)
Submit the GitHub URL to ReproAudit. The methodology alignment category will be skipped. Expected overall: **critical**.

### With manuscript (full audit)
Upload `manuscript/manuscript.pdf` when submitting. All 4 categories will be evaluated including methodology alignment mismatches. Expected overall: **critical**.
