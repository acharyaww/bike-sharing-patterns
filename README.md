# Bike-Sharing Usage Patterns

Weekday bike-share demand classification on the Seoul Bike-Sharing dataset (UCI Machine Learning Repository), comparing three classifiers: Logistic Regression, Bagging (Random Forest), and SVM (radial kernel).

## Result

| Model     | Accuracy | Misclassification rate |
|-----------|---------:|----------------------:|
| Logistic  | ~85%     | ~15%                  |
| **Bagging** | **95.13%** | **4.87%**          |
| SVM       | ~92%     | ~8%                   |

Bagging won. Tradeoff: logistic regression remains preferable when interpretability or training speed matters.

## Approach

1. Split the raw data into weekday vs weekend subsets.
2. Convert `Rented.Bike.Count` to a binary `countClass` using each subset's median (high vs low).
3. 80/20 train/test split with `caret::createDataPartition`.
4. Fit Logistic / Bagging / SVM on the same features; evaluate via `confusionMatrix`.

This repo reports on the **weekday** subset.

## Files

- `seoul-bike-data.Rmd` — full analysis (chunked, with prose)
- `SeoulBikeData.csv` — dataset (8,760 rows, 1 year hourly)

## How to run

```r
install.packages(c("dplyr", "tidyverse", "caret", "e1071", "randomForest"))
rmarkdown::render("seoul-bike-data.Rmd")
```

## Data source

Seoul Bike Sharing Demand Data Set — UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

## Notes

This is coursework — methodology emphasizes baseline comparison rather than tuning. SVM (`cost`, `gamma`) and Bagging (`mtry`) parameters used defaults to keep the comparison fair; tuning would likely close the gap.
