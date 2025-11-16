from cleaner import clean_all_files
from outstanding_players import calculate_player_metrics
from player_outliers import detect_player_outliers
from game_predictor import train_game_predictor, predict_specific_game

def run_full_pipeline():
    print("\n" + "="*60)
    print("NBA PREDICTION SYSTEM - COMPLETE ANALYSIS PIPELINE")
    print("="*60)
    
    print("\nDATA CLEANING...")
    clean_all_files()  
    
    print("\n1. FINDING OUTSTANDING PLAYERS...")
    outstanding_players = calculate_player_metrics() 
    
    print("\n2. DETECTING PLAYER OUTLIERS...")
    player_outliers = detect_player_outliers()
    
    print("\n3. TRAINING GAME OUTCOME PREDICTOR...")
    predictor, team_df = train_game_predictor()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    