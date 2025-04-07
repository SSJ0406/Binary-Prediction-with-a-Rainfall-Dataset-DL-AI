# 🌧️ Advanced Rainfall Prediction

## 📋 Overview
This repository contains an advanced ensemble model for binary rainfall prediction, built for the Kaggle Playground Series S4E12 competition. The solution integrates multiple ML models and custom meteorological feature engineering inspired by physical weather processes and top competition approaches.

The analysis was adapted to local file paths and enhanced with custom visualizations, weather pattern clustering, and a PyTorch neural network.

Key challenges tackled:
- **Imbalanced Classes**: Heavy skew towards rainfall occurrences (~76% positive class).
- **Meteorological Feature Engineering**: Derived humidity, pressure, and dew point interactions.
- **Clustering**: Used KMeans to identify weather patterns.
- **Advanced Ensembles**: Combined LightGBM, XGBoost, k-NN, and a PyTorch model into an adaptive ensemble.
- **Adaptive Visualization**: Cluster-specific model weighting with interpretability.

## 🎯 Key Results
- Top 41% on Kaggle private leaderboard.
- Cross-validated ensemble ROC-AUC: ~0.887
- Final blended model includes:
  - Neural Network (PyTorch)
  - LightGBM
  - XGBoost
  - k-Nearest Neighbors (k-NN)
- Visualization Highlights:
  - Cluster-wise weather patterns
  - Rainfall distribution and misclassifications
  - Feature effects and importances

## 🧪 PyTorch Neural Network Example
A key component of the ensemble is a simple but effective PyTorch neural network with Swish activation and batch normalization:

```python
class RainfallPredictionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.25):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
```

This model is trained using a WeightedRandomSampler to mitigate class imbalance and optimize AUC using ReduceLROnPlateau.

## 📊 Key Visualizations
All visualizations are saved to the `/visualizations` folder after execution. The generated visualizations include:
- `calibration_effect` - Analysis of prediction calibration showing the effect of probability adjustments

![image](https://github.com/user-attachments/assets/9cfad060-f959-4c5f-8734-8b45f65eee07)

- `cluster_weights` - Visualization of ensemble model weights across different weather patterns/clusters

![image](https://github.com/user-attachments/assets/15f554ab-46bd-4a92-8901-d17f1aac81fe)
  
- `feature_effects` - Feature response curves showing how individual variables impact rainfall predictions

![image](https://github.com/user-attachments/assets/d05cf47e-6e0f-47d3-b830-d8154f9ef410)
  
- `feature_importance` - Ranking and quantification of the most influential variables in the models

![image](https://github.com/user-attachments/assets/dc9a7472-67a8-4618-92cc-71b38cf41644)

- `model_comparison` - Performance metrics comparing AUC scores across different model types

![image](https://github.com/user-attachments/assets/c30f86f7-a5f0-41a6-a482-943e45ced227)

- `prediction_distribution` - Distribution analysis of rainfall predictions with probability thresholds

![image](https://github.com/user-attachments/assets/957c2400-9aa4-405e-809b-86c242d9c6bf)

- `weather_pattern_analysis` - KMeans clustering results showing different weather patterns and their characteristics

  ![image](https://github.com/user-attachments/assets/230e7efa-9b71-43be-ace0-8373f693ed30)

## 🛠️ What Was Done

### Feature Engineering
Calculated:
- Dew point spread, relative humidity, vapor pressure deficit
- Cloud × humidity interactions
- Wind vector components

Applied:
- Stratified 5-fold splits
- StandardScaler, MinMaxScaler, QuantileTransformer
- Clustered weather into 6 patterns using KMeans

### Models Used
- **LightGBM**: scale_pos_weight, tuned for regularization, with optimized parameters focused on reducing overfitting.
- **XGBoost**: scale_pos_weight, hist tree method, max depth 6, with gamma and alpha regularization.
- **k-NN**: n_neighbors=15, distance-based weighting, providing robust predictions complementary to tree-based models.
- **PyTorch Neural Network**: Deep learning model with a 2-layer MLP architecture featuring:
  - Batch normalization for better generalization
  - Swish/SiLU activation functions (outperforming ReLU)
  - Dropout layers (rate=0.25) for regularization
  - WeightedRandomSampler to address class imbalance
  - AdamW optimizer with weight decay
  - ReduceLROnPlateau learning rate scheduler

### Ensemble Techniques
- Simple weighted average
- Optimized ensemble weights (scipy minimize)
- Cluster-wise adaptive weighting based on weather pattern
- Isotonic calibration (optional)

## 🗂️ Repository Structure
```
📁 project-root/
├── visualizations/                   # Folder containing visualization outputs
├── rainfall_predictions.csv          # Final predictions file (18 KB)
├── test.csv                          # Test dataset (43 KB)
├── train.csv                         # Training dataset (132 KB)
├── rainfall_prediction.ipynb         # Main notebook (153 KB)
└── README.md
```

## 🚀 How to Run
Clone the repository:
```bash
git clone https://github.com/your-user/rainfall-prediction-kaggle.git
cd rainfall-prediction-kaggle
```

Install dependencies:
```bash
# Core libraries
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# Machine learning libraries (optional, but recommended)
pip install lightgbm xgboost torch

# Optional: for enhanced visualizations
pip install plotly
```

Run the notebook or execute the main() pipeline inside rainfall_prediction.ipynb.

## 🧠 Reflections and Learnings
- Combining meteorological domain knowledge with ML significantly improved interpretability and performance.
- k-NN, though simple, added robustness to the ensemble.
- Cluster-wise adaptive blending outperformed global weights.
- Visualization helped uncover model weaknesses in certain clusters.

## 🤝 Contributions
This project is not open for pull requests or edits. If you're inspired by the methodology or wish to build upon it, feel free to study the code — but please respect the license.

## 📜 License
This project's code and models are licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).
You may view, clone, and reference the code, but you are not permitted to:
- Use it for commercial purposes
- Modify the code or build derivative works
- Redistribute altered versions

⚠️ **Note**: The dataset used in this project is provided by Kaggle and is not the intellectual property of the author. Its usage is governed by the original data provider's terms.
