# Taxpayer Segmentation with Self-Organizing Maps

Unsupervised clustering of taxpayers registered in state active debt (*dívida ativa*), built as a Computational Intelligence project with data provided by the PGE (*Procuradoria-Geral do Estado*). The goal is to group debtors with similar profiles so collection strategies can be prioritized per group.

The pipeline combines a **Self-Organizing Map (SOM)** to project high-dimensional taxpayer profiles onto a 2D grid with **K-Means** to split that grid into groups, and lets a user look up which group a given taxpayer belongs to.

## Data

- ~1M records, 45 columns: debt status, collection phase and sphere, installment plans, lawsuits, protests, revenue and billed amounts per fiscal year, and several key dates.
- Personal and identifying fields (taxpayer code, company name, CPF/CNPJ, partners, accountant, address) arrive **already anonymized**.
- The dataset is not public and is **not included** in this repository.

## Pipeline

### 1. Data preparation

- **Cleaning** (rules validated with the PGE stakeholders)
  - `"-"` treated as null
  - drop suspended debts and records without fiscal year or debt type
  - drop write-off (*baixado*) taxpayers with no debt status
  - fill missing installment fields with sensible defaults
  - assert that each taxpayer code maps to a single CPF/CNPJ
- **Transformation:** decimal-comma strings to floats, text normalization, dates parsed and converted to Unix timestamps.
- **Encoding:** binary mapping for yes/no style features, one-hot encoding for the other categoricals.
- **Feature engineering:** yearly revenue (*arrecadado*) and billed (*faturado*) amounts are pivoted into one column per year.
- **Aggregation:** records are rolled up to one row per taxpayer, about 2.6k taxpayers with about 120 features.
- **Normalization:** `MinMaxScaler` to [0, 1].

About 500k records remain after cleaning.

### 2. Self-Organizing Map

Trained with [MiniSom](https://github.com/JustGlowing/minisom). Several grid sizes and hyperparameters were compared:

| Config | Grid | Learning rate | Sigma | Iterations | Resulting K |
|---|---|---|---|---|---|
| Best spread | 15×15 | 0.65 | 5 | 200 | 9 |
| Densest | 30×30 | 0.65 | 1 | 200 | 11 |
| Optimized | 100×100 | 0.65 | 5 | 500 | 11 |
| **Final** | **50×50** | **0.57** | **9** | **5000** | **4** |

The final model reaches a **quantization error of 0.08**. The notebook plots the neuron grid, the winning neurons per taxpayer and the U-Matrix (distance map) as a heatmap.

### 3. Clustering

K-Means runs on the coordinates of the winning neurons. K is chosen with:

- the **elbow method**, taking the point of maximum distance to the line joining the first and last WCSS values, and
- the **Silhouette Score** for K = 3 to 14.

Both point to **4 groups**, which are drawn over the SOM heatmap with each group's centroid and size.

### 4. Classifying a taxpayer

Given an (anonymized) taxpayer code, the notebook finds that taxpayer's winning neuron on the trained SOM and assigns it to the nearest K-Means centroid:

```
Digite o código do contribuinte: <anonymized code>
O neurônio ativado pertence ao grupo 4
```

## Tech stack

Python · pandas · NumPy · scikit-learn (K-Means, MinMaxScaler, silhouette score) · MiniSom · Matplotlib · Google Colab / Google Drive

## Running

1. Open `taxpayer-segmentation-som.ipynb` in Google Colab.
2. Point the `read_csv` / `to_csv` paths, currently under `/content/drive/Shareddrives/IA grupo 7/`, to your own copy of the data.
3. Run all cells. The first cell installs `minisom` and mounts Google Drive.

To run locally instead:

```bash
pip install pandas numpy matplotlib scikit-learn minisom jupyter
```

Then remove the `google.colab` import and the `drive.mount` call.

## Repository structure

```
.
├── taxpayer-segmentation-som.ipynb   # full pipeline: preparation, SOM, clustering, lookup
└── README.md
```

The last section of the notebook, *Códigos de exemplo*, holds earlier experiments: DBSCAN, a MiniSom example on the UCI Seeds dataset, and K-Means on a toy dataset.
