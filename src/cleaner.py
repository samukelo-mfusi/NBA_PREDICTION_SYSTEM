from loader import load_raw_file, save_clean

files = [
    "coaches_career.txt", "coaches_season.txt", "draft.txt",
    "player_allstar.txt", "player_playoffs.txt", "player_playoffs_career.txt",
    "player_regular_season.txt", "player_regular_season_career.txt",
    "players.txt", "team_season.txt", "teams.txt"
]

def clean_all_files():
    print("CLEANING DATA...")
    for f in files:
        try:
            df = load_raw_file(f)
            save_clean(df, f)
            print(f"Cleaned: {f}")
        except Exception as e:
            print(f"Failed: {f} — {e}")
