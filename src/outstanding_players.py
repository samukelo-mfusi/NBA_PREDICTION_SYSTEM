import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN = Path("data/cleaned")
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

def calculate_player_metrics():
    """Find outstanding players using career statistics"""
    
    print("...FINDING OUTSTANDING PLAYERS...")
    
    # Load data
    players_df = pd.read_csv(CLEAN / "players.csv")
    career_df = pd.read_csv(CLEAN / "player_regular_season_career.csv")
    
    print(f"Loaded {len(career_df)} player career records")
    print(f"Career data columns: {list(career_df.columns)}")
    
    # Merge to get player names - handle the merge carefully
    career_named = career_df.merge(
        players_df[['ilkid', 'firstname', 'lastname']], 
        on='ilkid', 
        how='left'
    )
    
    # Handle duplicate column names from merge
    if 'firstname_x' in career_named.columns and 'firstname_y' in career_named.columns:
        career_named['firstname'] = career_named['firstname_y'].fillna(career_named['firstname_x'])
        career_named['lastname'] = career_named['lastname_y'].fillna(career_named['lastname_x'])
        career_named = career_named.drop(['firstname_x', 'firstname_y', 'lastname_x', 'lastname_y'], axis=1)
    elif 'firstname_x' in career_named.columns:
        career_named['firstname'] = career_named['firstname_x']
        career_named['lastname'] = career_named['lastname_x']
        career_named = career_named.drop(['firstname_x', 'lastname_x'], axis=1)
    
    # Calculate per-game statistics
    career_named['ppg'] = career_named['pts'] / career_named['gp']
    career_named['rpg'] = career_named['reb'] / career_named['gp']
    career_named['apg'] = career_named['asts'] / career_named['gp']
    career_named['mpg'] = career_named['minutes'] / career_named['gp']
    
    # Advanced metrics
    career_named['efficiency'] = (
        career_named['pts'] + career_named['reb'] + career_named['asts'] + 
        career_named['stl'] + career_named['blk'] - career_named['turnover']
    ) / career_named['gp']
    
    # Player Impact Rating
    career_named['impact_rating'] = (
        career_named['ppg'] + career_named['rpg'] + career_named['apg'] +
        (career_named['stl'] / career_named['gp']) + (career_named['blk'] / career_named['gp'])
    )
    
    # Filter players with significant career (min 100 games)
    qualified_players = career_named[career_named['gp'] >= 100].copy()
    print(f"Players with 100+ games: {len(qualified_players)}")
    
    # Find outstanding players in different categories
    outstanding = {}
    
    # Top Scorers
    outstanding['top_scorers'] = qualified_players.nlargest(15, 'ppg')[
        ['firstname', 'lastname', 'ppg', 'pts', 'gp']
    ]
    
    # Most Efficient
    outstanding['most_efficient'] = qualified_players.nlargest(15, 'efficiency')[
        ['firstname', 'lastname', 'efficiency', 'ppg', 'rpg', 'apg']
    ]
    
    # Best All-Around (Impact)
    outstanding['best_all_around'] = qualified_players.nlargest(15, 'impact_rating')[
        ['firstname', 'lastname', 'impact_rating', 'ppg', 'rpg', 'apg']
    ]
    
    # Top Rebounders
    outstanding['top_rebounders'] = qualified_players.nlargest(15, 'rpg')[
        ['firstname', 'lastname', 'rpg', 'reb', 'gp']
    ]
    
    # Top Playmakers
    outstanding['top_playmakers'] = qualified_players.nlargest(15, 'apg')[
        ['firstname', 'lastname', 'apg', 'asts', 'gp']
    ]
    
    # Save detailed analysis as CSV files
    for category, data in outstanding.items():
        data.to_csv(CLEAN / f'outstanding_players_{category}.csv', index=False)
    
    # Create summary visualization
    create_outstanding_players_plot(outstanding)
    
    # Print top 5 in each category
    print("\nOUTSTANDING PLAYERS SUMMARY:")
    print("=" * 50)
    
    for category, data in outstanding.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for i, row in data.head(5).iterrows():
            player_name = f"{row['firstname']} {row['lastname']}"
            if 'ppg' in data.columns:
                print(f"  {player_name}: {row['ppg']:.1f} PPG")
            elif 'rpg' in data.columns:
                print(f"  {player_name}: {row['rpg']:.1f} RPG")
            elif 'apg' in data.columns:
                print(f"  {player_name}: {row['apg']:.1f} APG")
            elif 'efficiency' in data.columns:
                print(f"  {player_name}: {row['efficiency']:.1f} Efficiency")
            elif 'impact_rating' in data.columns:
                print(f"  {player_name}: {row['impact_rating']:.1f} Impact")
    
    return outstanding

def create_outstanding_players_plot(outstanding):
    """Create visualization of outstanding players"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NBA Outstanding Players Analysis', fontsize=16, fontweight='bold')
    
    # Top Scorers
    if 'top_scorers' in outstanding:
        data = outstanding['top_scorers'].head(10)
        axes[0,0].barh(range(len(data)), data['ppg'])
        axes[0,0].set_yticks(range(len(data)))
        axes[0,0].set_yticklabels([f"{row['firstname']} {row['lastname']}" for _, row in data.iterrows()], fontsize=8)
        axes[0,0].set_title('Top 10 Scorers (PPG)')
        axes[0,0].set_xlabel('Points Per Game')
    
    # Most Efficient
    if 'most_efficient' in outstanding:
        data = outstanding['most_efficient'].head(10)
        axes[0,1].barh(range(len(data)), data['efficiency'])
        axes[0,1].set_yticks(range(len(data)))
        axes[0,1].set_yticklabels([f"{row['firstname']} {row['lastname']}" for _, row in data.iterrows()], fontsize=8)
        axes[0,1].set_title('Top 10 Most Efficient')
        axes[0,1].set_xlabel('Efficiency Rating')
    
    # Best All-Around
    if 'best_all_around' in outstanding:
        data = outstanding['best_all_around'].head(10)
        axes[0,2].barh(range(len(data)), data['impact_rating'])
        axes[0,2].set_yticks(range(len(data)))
        axes[0,2].set_yticklabels([f"{row['firstname']} {row['lastname']}" for _, row in data.iterrows()], fontsize=8)
        axes[0,2].set_title('Top 10 All-Around Players')
        axes[0,2].set_xlabel('Impact Rating')
    
    # Top Rebounders
    if 'top_rebounders' in outstanding:
        data = outstanding['top_rebounders'].head(10)
        axes[1,0].barh(range(len(data)), data['rpg'])
        axes[1,0].set_yticks(range(len(data)))
        axes[1,0].set_yticklabels([f"{row['firstname']} {row['lastname']}" for _, row in data.iterrows()], fontsize=8)
        axes[1,0].set_title('Top 10 Rebounders (RPG)')
        axes[1,0].set_xlabel('Rebounds Per Game')
    
    # Top Playmakers
    if 'top_playmakers' in outstanding:
        data = outstanding['top_playmakers'].head(10)
        axes[1,1].barh(range(len(data)), data['apg'])
        axes[1,1].set_yticks(range(len(data)))
        axes[1,1].set_yticklabels([f"{row['firstname']} {row['lastname']}" for _, row in data.iterrows()], fontsize=8)
        axes[1,1].set_title('Top 10 Playmakers (APG)')
        axes[1,1].set_xlabel('Assists Per Game')
    
    # Remove empty subplot
    axes[1,2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(PLOTS / 'outstanding_players_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Outstanding players visualization saved!")