# NBA Basketball Performance Analytics and Game Outcome Prediction System

## The National Basketball Association (NBA) Prediction System

A comprehensive NBA analytics system that processes basketball statistics to identify outstanding players, detect statistical outliers, and predict game outcomes using machine learning.

## Features

- **Player Performance Analysis**: Identify top scorers, rebounders, playmakers, and efficient players
- **Statistical Outlier Detection**: Find unusual player profiles using Isolation Forest and statistical methods
- **Game Outcome Prediction**: Predict winners between teams with probability scores
- **Data Visualization**: Generate comprehensive plots and analysis reports

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Download or clone the project**
   ```bash
   # If using git:
   git clone https://github.com/samukelo-mfusi/NBA_PREDICTION_SYSTEM.git
   cd NBA_PREDICTION_SYSTEM
   
   # Or simply download and extract the ZIP file
   ```

2. **Install dependencies directly**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Run the complete analysis pipeline:**
```bash
python main.py
```

This will execute:
- Data cleaning and preparation
- Outstanding players identification  
- Player outlier detection
- Game prediction model training


## Outputs Generated

After running the pipeline, you'll get:

### Data Analysis
- `data/cleaned/` - Cleaned CSV files for all datasets
- `data/cleaned/outstanding_players_*.csv` - Top players by category

### Visualizations
- `plots/outstanding_players_comprehensive.png` - Top performers visualization
- `plots/player_outliers_analysis.png` - Outlier detection results
- `plots/feature_importance.png` - Model feature importance

### Models
- `models/game_predictor.pkl` - Trained prediction model

### Console Output
- Top 5 players in each category (scoring, efficiency, rebounding, playmaking)
- Statistical outliers with scores
- Model accuracy and feature importance
- Sample game predictions with probabilities

## Technical Details

### Algorithms Used
- **Random Forest Classifier** for game outcome prediction
- **Isolation Forest** for anomaly detection
- **Ensemble Methods** combining multiple outlier detection techniques
- **StandardScaler** for feature normalization

### Data Sources
- Historical NBA player statistics
- Team season performance data
- Player career records

### Key Metrics
- **Model Accuracy**: 85.71% on test data
- **Players Analyzed**: 3,759 career records
- **Team Seasons**: 1,187 seasons
- **Outliers Detected**: 296 players (7.9%)
