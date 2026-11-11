# 🛰️ Satellite Cloud Pattern Segmentation for Weather Data Exploration

## Using PCA, t-SNE, and K-Means Clustering

An interactive unsupervised machine learning application that explores satellite cloud patterns using dimensionality reduction and clustering techniques.

> **Project Type:** Unsupervised Machine Learning Research Project  
> **Domain:** Satellite Meteorology & Atmospheric Pattern Segmentation  
> **Methodology:** PCA, t-SNE, and K-Means Clustering  

---

## 📋 Abstract

This project applies unsupervised machine learning techniques—**Principal Component Analysis (PCA)**, **t-Distributed Stochastic Neighbor Embedding (t-SNE)**, and **K-Means Clustering**—to explore naturally occurring patterns in satellite-derived cloud and weather observation data. The application identifies potential cloud-pattern regimes without imposing predefined meteorological categories, providing an interactive Streamlit dashboard for exploration and analysis.

## 🎯 Problem Statement

Satellite-derived cloud data provides a rich source of atmospheric information, but its high dimensionality makes manual analysis challenging. This project investigates:

1. Can we reduce dimensionality of weather features while preserving information?
2. Do naturally occurring groups emerge in the feature space?
3. Can these groups be interpreted as potential weather/cloud-pattern regimes?

## 📊 Dataset

### Source
- **Name:** Cloud Weather Observations (based on NOAA LCD distributions)
- **Basis:** NOAA Local Climatological Data (LCD) statistical distributions
- **License:** Public Domain (U.S. Government Work reference data)
- **Coverage:** 15 Indian weather stations, Jan 2023 – Dec 2024

### Statistics
| Metric | Value |
|--------|-------|
| Records | 15,075 |
| Total Features | 22 |
| Numerical Features | 14 |
| Categorical Features | 4 |
| Stations | 15 |
| Time Span | 2 years |
| Missing Values | ~3% |

### Key Features
- `temperature` — Air temperature (°C)
- `dew_point` — Dew point temperature (°C)
- `relative_humidity` — Relative humidity (%)
- `wind_speed` — Wind speed (km/h)
- `wind_direction` — Wind direction (degrees)
- `station_pressure` — Atmospheric pressure (hPa)
- `sea_level_pressure` — Sea-level pressure (hPa)
- `visibility` — Horizontal visibility (km)
- `cloud_ceiling` — Cloud base height (meters)
- `cloud_cover_fraction` — Sky coverage (0-1)
- `precipitation` — Precipitation amount (mm)
- `wet_bulb_temperature` — Wet bulb temperature (°C)

### Data Source Note
INSAT-3D/3DR data from ISRO MOSDAC is available in HDF5/NetCDF format but requires registration and conversion. This project uses a representative dataset based on NOAA LCD distributions. The pipeline is designed to accept real satellite data by placing a CSV in `data/raw/`.

## 🛠️ Technologies

| Category | Technologies |
|----------|-------------|
| Language | Python 3.12 |
| ML | Scikit-learn, NumPy, SciPy |
| Data | Pandas |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Persistence | Joblib |

## 🔬 Methodology

### 1. Data Preprocessing
- Duplicate detection and removal
- Missing value imputation (median strategy for numerical, mode for categorical)
- Feature selection (14 numerical features)
- StandardScaler normalization (mean=0, std=1)
- Outlier detection (IQR method — outliers retained for weather data)

### 2. Principal Component Analysis (PCA)
- Linear dimensionality reduction
- Explained variance analysis and scree plots
- Component loading analysis
- Optimal component selection (95% variance threshold)
- 2D and 3D visualization

### 3. t-SNE Analysis
- Non-linear 2D projection for visualization
- Parameter experimentation (perplexity, learning rate, iterations)
- PCA preprocessing before t-SNE
- Sampling for computational efficiency

### 4. K-Means Clustering
- Applied on PCA-reduced features
- Elbow Method for optimal K selection
- Silhouette Score evaluation
- Cluster centroid analysis and profiling
- Radar charts for cluster comparison

## 🏗️ Project Structure

```
satellite-cloud-segmentation/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration and parameters
├── prepare_dataset.py        # Dataset generation script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/
│   ├── raw/                  # Raw dataset CSV
│   └── processed/            # Processed data
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and validation
│   ├── preprocessing.py      # Preprocessing pipeline
│   ├── feature_analysis.py   # Statistical analysis
│   ├── pca_analysis.py       # PCA implementation
│   ├── tsne_analysis.py      # t-SNE implementation
│   ├── clustering.py         # K-Means clustering
│   └── evaluation.py         # Evaluation and comparison
│
├── views/
│   ├── __init__.py
│   ├── overview.py           # Page 1: Project overview
│   ├── data_exploration.py   # Page 2: Data exploration
│   ├── preprocessing_page.py # Page 3: Preprocessing
│   ├── pca_page.py           # Page 4: PCA analysis
│   ├── tsne_page.py          # Page 5: t-SNE analysis
│   ├── clustering_page.py    # Page 6: K-Means clustering
│   ├── comparison.py         # Page 7: Comparative analysis
│   └── conclusions.py        # Page 8: Conclusions & report
│
├── outputs/
│   ├── plots/
│   └── reports/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
└── assets/
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Steps

1. **Navigate to the project directory:**
   ```bash
   cd satellite-cloud-segmentation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the dataset (if not already present):**
   ```bash
   python prepare_dataset.py
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser:**
   The application will open automatically at `http://localhost:8501`

## 📱 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Project introduction, objectives, dataset summary |
| 🔍 Data Exploration | Raw data, distributions, correlations, geographic/temporal analysis |
| ⚙️ Preprocessing | Missing values, feature selection, scaling, outlier detection |
| 📊 PCA Analysis | Variance analysis, scree plot, loadings, 2D/3D scatter |
| 🌀 t-SNE Analysis | Interactive t-SNE with parameter controls |
| 🎯 K-Means Clustering | Elbow curve, silhouette, cluster visualization, centroid analysis |
| 📈 Comparative Analysis | PCA vs t-SNE comparison, clustering across K values |
| 📋 Conclusions & Report | Key findings, limitations, ethics, downloadable report |

## ⚠️ Limitations

1. This is an **exploratory analysis** — clusters are mathematical groupings, not confirmed cloud types.
2. K-Means assumes spherical clusters, which may not match actual weather pattern geometry.
3. t-SNE distances between clusters are **not meaningful**.
4. Independent meteorological validation is required for cloud type interpretations.
5. The dataset is representative, not actual observed data.

## 🔒 Ethical Considerations

- This application is for **educational/exploratory purposes only**, NOT operational weather forecasting.
- Cluster assignments should never be used for safety-critical weather decisions.
- The dataset may underrepresent certain geographical regions or weather conditions.

## 📚 References

1. Jolliffe, I. T., & Cadima, J. (2016). Principal Component Analysis: A Review. *Phil. Trans. R. Soc. A*, 374(2065).
2. van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*, 9, 2579–2605.
3. Lloyd, S. P. (1982). Least Squares Quantization in PCM. *IEEE Trans. Info. Theory*, 28(2).
4. Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid. *J. Comp. Appl. Math.*, 20.
5. NOAA NCEI. Local Climatological Data. https://www.ncei.noaa.gov/data/local-climatological-data/
6. ISRO MOSDAC. INSAT-3D Satellite Data. https://mosdac.gov.in/
7. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12.
8. Wattenberg, M., et al. (2016). How to Use t-SNE Effectively. *Distill*.

## 📝 License

This project is open-source and intended for educational, research, and scientific exploration purposes.

---

*Built with Python, Scikit-learn, and Streamlit*

