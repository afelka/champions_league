import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
URL = "https://en.wikipedia.org/wiki/2024%E2%80%9325_UEFA_Champions_League_league_phase"

HEADERS = {
    "User-Agent": (
        "UCLPotScraper/1.0 "
        "(contact: your.email@example.com) "
        "Educational / non-commercial use"
    )
}

# -----------------------------
# Fetch page (policy compliant)
# -----------------------------
response = requests.get(URL, headers=HEADERS, timeout=30)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

teams = []

# -----------------------------
# Find pot tables via captions
# -----------------------------
for caption in soup.find_all("caption"):
    pot_text = caption.get_text(strip=True)

    if not pot_text.startswith("Pot"):
        continue

    table = caption.find_parent("table")

    # Skip header row
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 3:
            continue

        # -----------------------------
        # Extract team name (ignore flags)
        # -----------------------------
        team = None
        for a in cols[0].find_all("a", href=True):
            text = a.get_text(strip=True)
            if text:
                team = text
                break

        if not team:
            continue

        # -----------------------------
        # Extract coefficient
        # -----------------------------
        coeff_text = cols[-1].get_text(strip=True)
        try:
            coefficient = float(coeff_text)
        except ValueError:
            coefficient = None

        teams.append({
            "pot": pot_text,
            "team": team,
            "coefficient": coefficient
        })

# Be polite if expanding to multiple pages
time.sleep(1)

# -----------------------------
# Output
# -----------------------------
for r in teams:
    print(r)

teams_df = pd.DataFrame(teams)

results = []

for table in soup.find_all("table", class_="sports-series"):
    matchday_name = table.find("caption").get_text(strip=True)
    for row in table.find_all("tr")[1:]:  # skip header
        cols = row.find_all("td")
        if len(cols) < 3:
            continue
        home = cols[0].get_text(strip=True)
        score = cols[1].get_text(strip=True).replace("–", "-")
        away = cols[2].get_text(strip=True)

        results.append({
            "matchday": matchday_name,
            "home": home,
            "score": score,
            "away": away
        })

for r in results:
    print(r)

results_df = pd.DataFrame(results)

name_map = {
    'ajax': 'afc ajax',
    'union saint-gilloise': 'union saint-gilloise',  # normalize weird hyphens/non-breaking spaces
    # add more mappings as needed
}

def normalize_team(name):
    if pd.isna(name):
        return ''
    # replace non-breaking spaces and special hyphens
    name = name.replace('\xa0', ' ').replace('‑', '-')
    name = name.strip().lower()
    # apply mapping if exists
    return name_map.get(name, name)

# Create normalized columns
results_df['home_norm'] = results_df['home'].apply(normalize_team)
results_df['away_norm'] = results_df['away'].apply(normalize_team)
teams_df['team_norm'] = teams_df['team'].apply(normalize_team)

# Merge pot info for home team
results_df = results_df.merge(
    teams_df[['team_norm', 'pot']],
    how='left',
    left_on='home_norm',
    right_on='team_norm'
).rename(columns={'pot': 'home_pot'}).drop(columns=['team_norm'])

# Merge pot info for away team
results_df = results_df.merge(
    teams_df[['team_norm', 'pot']],
    how='left',
    left_on='away_norm',
    right_on='team_norm'
).rename(columns={'pot': 'away_pot'}).drop(columns=['team_norm'])

# Drop temporary normalized columns if you want
results_df = results_df.drop(columns=['home_norm', 'away_norm'])

print(results_df.head())

# Keep only played games
played = results_df[results_df['score'].str.contains('-')].copy()

# Compute home result
def get_home_result(score):
    home_goals, away_goals = map(int, score.split('-'))
    if home_goals > away_goals:
        return 'win'
    elif home_goals < away_goals:
        return 'loss'
    else:
        return 'draw'

played['home_result'] = played['score'].apply(get_home_result)

# Prepare home perspective
home_df = played[['home_pot','away_pot','home_result']].copy()
home_df.rename(columns={'home_pot':'pot','away_pot':'opponent','home_result':'result'}, inplace=True)

# Prepare away perspective and invert result
away_df = played[['away_pot','home_pot','home_result']].copy()
away_df.rename(columns={'away_pot':'pot','home_pot':'opponent','home_result':'result'}, inplace=True)
away_df['result'] = away_df['result'].map({'win':'loss','loss':'win','draw':'draw'})

# Combine home and away
combined = pd.concat([home_df, away_df], ignore_index=True)

# Compute counts of each result per pot vs opponent
pot_results = combined.groupby(['pot','opponent','result']).size().unstack(fill_value=0)

# Ensure all pots and opponents are included (even if zero)
pots = sorted(combined['pot'].unique())
pot_results = pot_results.reindex(pd.MultiIndex.from_product([pots, pots], names=['pot','opponent']), fill_value=0)

# Plot
n = len(pots)
fig, axes = plt.subplots(n, n, figsize=(4*n, 4*n), sharex=False, sharey=True)

max_val = pot_results[['win','draw','loss']].to_numpy().max()
scale = max_val + 2

for i, pot in enumerate(pots):
    for j, opponent in enumerate(pots):
        ax = axes[i, j]
        values = pot_results.loc[(pot, opponent)][['win','draw','loss']]
        bars = ax.bar([0,1,2], values, color=['green','gray','red'])
        
        # Add numbers on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, str(val), ha='center', va='bottom')
        
        ax.set_ylim(0, scale)
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['W','D','L'])  # force W/D/L on every plot

        if j == 0:
            ax.set_ylabel(pot)
        if i == n-1:
            ax.set_xlabel(opponent)
        ax.set_title(f'{pot} vs {opponent}', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle('Champions League 2024-2025 Pot vs Pot Results', fontsize=16)
plt.savefig('./images/pots_vs_pots_results_2024-2025.png', dpi=300)
plt.show()