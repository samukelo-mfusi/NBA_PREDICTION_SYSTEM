import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

CLEAN = Path("data/cleaned")
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

def detect_player_outliers():
    """Detect statistical outliers in player performance"""
    
    print("...DETECTING PLAYER OUTLIERS...")
    
    # Load data
    players_df = pd.read_csv(CLEAN / "players.csv")
    career_df = pd.read_csv(CLEAN / "player_regular_season_career.csv")
    
    # Merge for player names
    player_data = career_df.merge(
        players_df[['ilkid', 'firstname', 'lastname']], 
        on='ilkid', 
        how='left'
    )
    
    # Handle column names after merge
    if 'firstname_x' in player_data.columns and 'firstname_y' in player_data.columns:
        player_data['firstname'] = player_data['firstname_y'].fillna(player_data['firstname_x'])
        player_data['lastname'] = player_data['lastname_y'].fillna(player_data['lastname_x'])
        player_data = player_data.drop(['firstname_x', 'firstname_y', 'lastname_x', 'lastname_y'], axis=1)
    
    print(f"Analyzing {len(player_data)} players for outliers...")
    
    # Calculate per-game statistics
    player_data['ppg'] = player_data['pts'] / player_data['gp']
    player_data['rpg'] = player_data['reb'] / player_data['gp']
    player_data['apg'] = player_data['asts'] / player_data['gp']
    player_data['mpg'] = player_data['minutes'] / player_data['gp']
    player_data['spg'] = player_data['stl'] / player_data['gp']
    player_data['bpg'] = player_data['blk'] / player_data['gp']
    
    # Features for outlier detection
    features = ['ppg', 'rpg', 'apg', 'mpg', 'spg', 'bpg', 'turnover']
    available_features = [f for f in features if f in player_data.columns]
    
    print(f"Using features: {available_features}")
    
    # Prepare feature matrix
    X = player_data[available_features].fillna(0)
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Multiple outlier detection methods
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_scores = iso_forest.fit_predict(X)
    player_data['outlier_iso'] = (iso_scores == -1).astype(int)
    
    # Z-score method
    from scipy import stats
    z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
    player_data['outlier_z'] = (z_scores.max(axis=1) > 3).astype(int)
    
    # Statistical outliers (beyond 3 standard deviations)
    statistical_outliers = []
    for feature in available_features:
        mean = player_data[feature].mean()
        std = player_data[feature].std()
        outliers = np.abs(player_data[feature] - mean) > (3 * std)
        statistical_outliers.append(outliers)
    
    player_data['outlier_stat'] = np.any(statistical_outliers, axis=0).astype(int)
    
    # Combined outlier score
    player_data['outlier_score'] = (
        player_data['outlier_iso'] + 
        player_data['outlier_z'] + 
        player_data['outlier_stat']
    )
    
    # Identify significant outliers (score >= 2)
    significant_outliers = player_data[player_data['outlier_score'] >= 2].copy()
    significant_outliers['player_name'] = significant_outliers['firstname'] + ' ' + significant_outliers['lastname']
    
    print(f"\nFound {len(significant_outliers)} significant outliers")
    
    # Save results
    outlier_cols = ['player_name', 'ppg', 'rpg', 'apg', 'outlier_score', 'outlier_iso', 'outlier_z', 'outlier_stat']
    significant_outliers[outlier_cols].to_csv(CLEAN / 'player_outliers_detected.csv', index=False)
    
    # Print top outliers
    print("\nTOP STATISTICAL OUTLIERS:")
    print("=" * 60)
    for _, player in significant_outliers.nlargest(10, 'outlier_score').iterrows():
        print(f"{player['player_name']:20} | Score: {player['outlier_score']} | PPG: {player['ppg']:.1f} | RPG: {player['rpg']:.1f} | APG: {player['apg']:.1f}")
    
    # Create visualizations
    create_outlier_visualizations(player_data, significant_outliers, available_features)
    
    return significant_outliers

def create_outlier_visualizations(player_data, outliers, features):
    """Create comprehensive outlier visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('NBA Player Performance Outlier Detection', fontsize=16, fontweight='bold')
    
    # 1. Outlier score distribution
    outlier_counts = player_data['outlier_score'].value_counts().sort_index()
    axes[0,0].bar(outlier_counts.index, outlier_counts.values, color='skyblue')
    axes[0,0].set_title('Distribution of Outlier Scores')
    axes[0,0].set_xlabel('Outlier Score')
    axes[0,0].set_ylabel('Number of Players')
    
    # 2. PPG vs RPG with outliers highlighted
    colors = ['blue' if score == 0 else 'orange' if score == 1 else 'red' for score in player_data['outlier_score']]
    axes[0,1].scatter(player_data['ppg'], player_data['rpg'], c=colors, alpha=0.6)
    axes[0,1].set_title('Points vs Rebounds Per Game\n(Blue: Normal, Orange: Mild, Red: Strong Outliers)')
    axes[0,1].set_xlabel('Points Per Game')
    axes[0,1].set_ylabel('Rebounds Per Game')
    
    # 3. Feature distribution with outliers
    feature_data = player_data[features[:3]].melt()  # First 3 features
    sns.boxplot(data=feature_data, x='value', y='variable', ax=axes[1,0])
    axes[1,0].set_title('Distribution of Key Player Statistics')
    axes[1,0].set_xlabel('Value')
    
    # 4. Top outliers
    if len(outliers) > 0:
        top_outliers = outliers.nlargest(8, 'outlier_score')
        axes[1,1].barh(
            [name[:18] + '...' if len(name) > 18 else name for name in top_outliers['player_name']],
            top_outliers['outlier_score'],
            color='red'
        )
        axes[1,1].set_title('Top Statistical Outliers')
        axes[1,1].set_xlabel('Outlier Score')
    
    plt.tight_layout()
    plt.savefig(PLOTS / 'player_outliers_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Player outlier analysis completed and visualizations saved!")