import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

CLEAN = Path("data/cleaned")
MODELS = Path("models")
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

class GamePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.teams_info = None
    
    def prepare_game_data(self):
        """Prepare data for game outcome prediction"""
        
        print("Preparing game outcome prediction data...")
        
        # Load team data
        team_df = pd.read_csv(CLEAN / "team_season.csv")
        teams_info = pd.read_csv(CLEAN / "teams.csv")
        
        print(f"Team data shape: {team_df.shape}")
        print(f"Available columns: {list(team_df.columns)}")
        
        # Create target: winning season (above 0.500 win percentage)
        team_df['win_pct'] = team_df['won'] / (team_df['won'] + team_df['lost'])
        team_df['target'] = (team_df['win_pct'] > 0.5).astype(int)
        
        print(f"Target distribution: {team_df['target'].value_counts().to_dict()}")
        
        # Select features (exclude target-related columns)
        exclude_cols = ['target', 'win_pct', 'won', 'lost', 'team', 'leag', 'year']
        feature_cols = [col for col in team_df.columns if col not in exclude_cols and team_df[col].dtype in [np.int64, np.float64]]
        
        self.feature_columns = feature_cols
        self.teams_info = teams_info
        
        X = team_df[feature_cols].fillna(0)
        y = team_df['target']
        
        print(f"Using {len(feature_cols)} features for prediction")
        
        return X, y, team_df
    
    def train_model(self, X, y):
        """Train the game prediction model"""
        
        print("Training game outcome predictor...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return train_acc, test_acc, feature_importance
    
    def predict_game(self, team1_name, team2_name, team_df):
        """Predict outcome between two specific teams"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Find teams
        team1 = self._find_team(team1_name, team_df)
        team2 = self._find_team(team2_name, team_df)
        
        if team1 is None or team2 is None:
            return None
        
        # Prepare features
        team1_features = team1[self.feature_columns].fillna(0).values.reshape(1, -1)
        team2_features = team2[self.feature_columns].fillna(0).values.reshape(1, -1)
        
        # Scale features
        team1_scaled = self.scaler.transform(team1_features)
        team2_scaled = self.scaler.transform(team2_features)
        
        # Get probabilities
        team1_prob = self.model.predict_proba(team1_scaled)[0][1]
        team2_prob = self.model.predict_proba(team2_scaled)[0][1]
        
        # Determine winner
        if team1_prob > team2_prob:
            winner = team1_name
            winner_prob = team1_prob
            loser_prob = team2_prob
        else:
            winner = team2_name
            winner_prob = team2_prob
            loser_prob = team1_prob
        
        confidence = abs(team1_prob - team2_prob)
        
        return {
            'winner': winner,
            'winner_prob': winner_prob,
            'loser_prob': loser_prob,
            'confidence': confidence,
            'team1_prob': team1_prob,
            'team2_prob': team2_prob
        }
    
    def _find_team(self, team_name, team_df):
        """Find team data by name"""
        team_name_lower = team_name.lower()
        
        # Try exact match first
        matches = team_df[team_df['team'].str.lower() == team_name_lower]
        if len(matches) > 0:
            return matches.nlargest(1, 'year').iloc[0]
        
        # Try partial match
        matches = team_df[team_df['team'].str.lower().str.contains(team_name_lower, na=False)]
        if len(matches) > 0:
            return matches.nlargest(1, 'year').iloc[0]
        
        # Try team info lookup
        if self.teams_info is not None:
            location_matches = self.teams_info[self.teams_info['location'].str.lower().str.contains(team_name_lower, na=False)]
            if len(location_matches) > 0:
                team_abbr = location_matches.iloc[0]['team']
                team_matches = team_df[team_df['team'] == team_abbr]
                if len(team_matches) > 0:
                    return team_matches.nlargest(1, 'year').iloc[0]
            
            name_matches = self.teams_info[self.teams_info['name'].str.lower().str.contains(team_name_lower, na=False)]
            if len(name_matches) > 0:
                team_abbr = name_matches.iloc[0]['team']
                team_matches = team_df[team_df['team'] == team_abbr]
                if len(team_matches) > 0:
                    return team_matches.nlargest(1, 'year').iloc[0]
        
        print(f"Team '{team_name}' not found. Available teams: {team_df['team'].unique()[:10]}")
        return None
    
    def save_model(self):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'teams_info': self.teams_info
        }
        joblib.dump(model_data, MODELS / 'game_predictor.pkl')
        print("Game predictor model saved!")
    
    def load_model(self):
        """Load trained model"""
        try:
            model_data = joblib.load(MODELS / 'game_predictor.pkl')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.teams_info = model_data['teams_info']
            print("Game predictor model loaded!")
            return True
        except:
            print("No saved model found.")
            return False

def train_game_predictor():
    """Train the game outcome predictor"""
    
    predictor = GamePredictor()
    X, y, team_df = predictor.prepare_game_data()
    train_acc, test_acc, feature_importance = predictor.train_model(X, y)
    predictor.save_model()
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title('Feature Importance for Game Outcome Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(PLOTS / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return predictor, team_df

def predict_specific_game(team1_name, team2_name):
    """Predict outcome between two specific teams"""
    
    predictor = GamePredictor()
    if not predictor.load_model():
        print("Please train the model first.")
        return
    
    team_df = pd.read_csv(CLEAN / "team_season.csv")
    
    result = predictor.predict_game(team1_name, team2_name, team_df)
    
    if result:
        print(f"\nGAME PREDICTION: {team1_name} vs {team2_name}")
        print("=" * 50)
        print(f"Predicted Winner: {result['winner']}")
        print(f"{team1_name} Win Probability: {result['team1_prob']:.3f}")
        print(f"{team2_name} Win Probability: {result['team2_prob']:.3f}")
        print(f"Prediction Confidence: {result['confidence']:.3f}")
        print("=" * 50)
        
        return result
    else:
        print("Could not make prediction. Check team names.")
        return None