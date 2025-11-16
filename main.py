import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import run_full_pipeline
from game_predictor import predict_specific_game

if __name__ == "__main__":
    run_full_pipeline()
    
    predict_specific_game("Celtics", "Knicks")
    predict_specific_game("Bulls", "Knicks")
    predict_specific_game("Heat", "Celtics") 
