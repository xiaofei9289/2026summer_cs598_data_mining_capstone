# Reproducibility and task versions

## Inspect the exported results

`results/model_comparison.csv` contains three individual fold F1 scores per model. The displayed 0.601 is their arithmetic mean, not the macro-F1 of pooled OOF predictions. `std_f1` uses `ddof=0` and is not a confidence interval.

`results/oof_confusion_matrix.csv` and `results/oof_classification_report.csv` describe pooled out-of-fold predictions. They allow reconstruction of risk-class precision, recall, and F1.

## Rebuild the main predictive cohort

1. Acquire the original Yelp review/business data and record the dataset version.
2. Apply the July 1, 2013 historical/future cutoff.
3. Apply minimum support: six historical, five future reviews.
4. Label risk when historical mean minus future mean is at least 0.75 and future mean is at most 3.5.
5. Build historical text and metadata only; future ratings/counts must not become model features.
6. Use the recorded three-fold stratified split and fit text/numeric preprocessing within each training fold.
7. Export fold scores and all out-of-fold predictions; compare against the majority dummy.

Recorded main-model parameters: word TF-IDF 1–2 grams, `min_df=2`, `max_features=5000`, English stop words, sublinear TF; class-balanced LR with `C=0.5`, `max_iter=1000`, random seed 42.

## Source notebooks to synchronize

The repository includes these notebook entry points (outputs removed):

| Task | Source file |
|---|---|
| Original Mexican topic study | `notebooks/task1_student_v20.ipynb` |
| Cuisine representations | `notebooks/task2_student_v20.ipynb` |
| Dish vocabulary | `notebooks/task3_student_v21.ipynb` |
| Dish evidence/ranking | `notebooks/task4_student_v21.ipynb` |
| Temporal analysis | `notebooks/task5_student_v21.ipynb` |
| Risk prediction | `notebooks/task6_student_v21.ipynb` |

The later Task1 Chinese analysis is a different version. Keep one canonical version per result and label alternatives, rather than mixing Mexican and Chinese counts.

## Release requirements

Record the source commit, notebook order, Python/package versions, dataset preparation, paths/configuration, and expected output schemas. Provide a small synthetic input for a quick pipeline demonstration if the original data are not bundled; label synthetic outputs as examples, not measured results.

The six completed notebooks are included with outputs, execution counts, and machine metadata removed; their cell sources are unchanged. Dataset preparation and the environment must be supplied before an end-to-end rerun. This release does not claim a newly tested training or notebook execution.
