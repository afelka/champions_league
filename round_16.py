import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# Range covering between 2000-01 and 2023-24 seasons
seasons = [(str(y), str(y+1)[2:]) for y in range(2000, 2024)]
all_teams = []
HEADERS = {"User-Agent": "UCL_History_Project/1.0"}

for start_yr, end_yr in seasons:
    season_str = f"{start_yr}–{end_yr}"
    year_int = int(start_yr)
    found_in_season = False
    
    # URL Logic
    urls_to_try = []
    if year_int < 2003:
        urls_to_try.append(f"https://en.wikipedia.org/wiki/{season_str}_UEFA_Champions_League_second_group_stage")
    else:
        urls_to_try.append(f"https://en.wikipedia.org/wiki/{season_str}_UEFA_Champions_League_knockout_phase")
        urls_to_try.append(f"https://en.wikipedia.org/wiki/{season_str}_UEFA_Champions_League_knockout_stage")
        urls_to_try.append(f"https://en.wikipedia.org/wiki/{season_str}_UEFA_Champions_League")

    for url in urls_to_try:
        if found_in_season: break
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            if res.status_code != 200: continue
            soup = BeautifulSoup(res.text, "html.parser")

            # --- CLASSIC FORMAT LOGIC (2003-2023) ---
            if year_int >= 2003:
                for table in soup.find_all("table", class_="wikitable"):
                    header_text = table.get_text().lower()
                    if "winners" in header_text and "runner" in header_text:
                        for row in table.find_all("tr"):
                            cols = row.find_all(["td", "th"])
                            if len(cols) >= 3:
                                if "winner" in cols[1].get_text().lower(): continue
                                for i in [1, 2]:
                                    flag = cols[i].find("span", class_="flagicon")
                                    country = flag.find("img")["alt"] if flag else None
                                    team_link = next((a for a in cols[i].find_all("a") if not a.find_parent("span", class_="flagicon")), None)
                                    if team_link:
                                        all_teams.append({"season": season_str, "country": country, "team": team_link.get_text(strip=True), "rank": "Winner" if i==1 else "Runner-up"})
                                        found_in_season = True
                        if found_in_season: break

            # --- VINTAGE FORMAT LOGIC (2000-2002) ---
            else:
                for table in soup.find_all("table", class_="wikitable"):
                    if "Team" in table.get_text():
                        for row in table.find_all("tr")[1:]:
                            cols = row.find_all("td")
                            if len(cols) >= 2:
                                flag = cols[0].find("span", class_="flagicon")
                                country = flag.find("img")["alt"] if flag else None
                                cell_copy = BeautifulSoup(str(cols[0]), "html.parser")
                                for s in cell_copy.find_all("span", class_="flagicon"): s.decompose()
                                team = cell_copy.get_text(strip=True)
                                if team and country:
                                    all_teams.append({"season": season_str, "country": country, "team": team, "rank": "N/A"})
                                    found_in_season = True
        except: continue

    print(f"{'✅' if found_in_season else '⚠️'} Processed {season_str}")
    time.sleep(1)

df_final = pd.DataFrame(all_teams).drop_duplicates()

# Data for 2024–25 and 2025–26 UCL Round of 16
manual_data = [
    # --- 2024–25 SEASON ---
    {"season": "2024–25", "country": "England", "team": "Liverpool", "rank": "Top 8 (Direct)"},
    {"season": "2024–25", "country": "Spain", "team": "Barcelona", "rank": "Top 8 (Direct)"},
    {"season": "2024–25", "country": "England", "team": "Arsenal", "rank": "Top 8 (Direct)"},
    {"season": "2024–25", "country": "Italy", "team": "Inter Milan", "rank": "Top 8 (Direct)"},
    {"season": "2024–25", "country": "Spain", "team": "Atlético Madrid", "rank": "Top 8 (Direct)"},
    {"season": "2024–25", "country": "Germany", "team": "Bayer Leverkusen", "rank": "Top 8 (Direct)"},
    {"season": "2024–25", "country": "France", "team": "Lille", "rank": "Top 8 (Direct)"},
    {"season": "2024–25", "country": "England", "team": "Aston Villa", "rank": "Top 8 (Direct)"},
    {"season": "2024–25", "country": "Spain", "team": "Real Madrid", "rank": "Play-off Winner"},
    {"season": "2024–25", "country": "France", "team": "Paris Saint-Germain", "rank": "Play-off Winner"},
    {"season": "2024–25", "country": "Germany", "team": "Bayern Munich", "rank": "Play-off Winner"},
    {"season": "2024–25", "country": "Germany", "team": "Borussia Dortmund", "rank": "Play-off Winner"},
    {"season": "2024–25", "country": "Portugal", "team": "Benfica", "rank": "Play-off Winner"},
    {"season": "2024–25", "country": "Netherlands", "team": "PSV Eindhoven", "rank": "Play-off Winner"},
    {"season": "2024–25", "country": "Netherlands", "team": "Feyenoord", "rank": "Play-off Winner"},
    {"season": "2024–25", "country": "Belgium", "team": "Club Brugge", "rank": "Play-off Winner"},

    # --- 2025–26 SEASON ---
    {"season": "2025–26", "country": "England", "team": "Arsenal", "rank": "Top 8 (Direct)"},
    {"season": "2025–26", "country": "Germany", "team": "Bayern Munich", "rank": "Top 8 (Direct)"},
    {"season": "2025–26", "country": "England", "team": "Liverpool", "rank": "Top 8 (Direct)"},
    {"season": "2025–26", "country": "England", "team": "Tottenham Hotspur", "rank": "Top 8 (Direct)"},
    {"season": "2025–26", "country": "Spain", "team": "Barcelona", "rank": "Top 8 (Direct)"},
    {"season": "2025–26", "country": "England", "team": "Chelsea", "rank": "Top 8 (Direct)"},
    {"season": "2025–26", "country": "Portugal", "team": "Sporting CP", "rank": "Top 8 (Direct)"},
    {"season": "2025–26", "country": "England", "team": "Manchester City", "rank": "Top 8 (Direct)"},
    {"season": "2025–26", "country": "Spain", "team": "Real Madrid", "rank": "Play-off Winner"},
    {"season": "2025–26", "country": "France", "team": "Paris Saint-Germain", "rank": "Play-off Winner"},
    {"season": "2025–26", "country": "England", "team": "Newcastle United", "rank": "Play-off Winner"},
    {"season": "2025–26", "country": "Spain", "team": "Atlético Madrid", "rank": "Play-off Winner"},
    {"season": "2025–26", "country": "Italy", "team": "Atalanta", "rank": "Play-off Winner"},
    {"season": "2025–26", "country": "Germany", "team": "Bayer Leverkusen", "rank": "Play-off Winner"},
    {"season": "2025–26", "country": "Turkey", "team": "Galatasaray", "rank": "Play-off Winner"},
    {"season": "2025–26", "country": "Norway", "team": "Bodø/Glimt", "rank": "Play-off Winner"}
]

# Create the DataFrame
df_new_era = pd.DataFrame(manual_data)

# To merge this with your existing 'df_final' (scraped data):
df_complete = pd.concat([df_final, df_new_era], ignore_index=True)

df_complete[["season", "country", "team"]].to_csv("teams_in_round_16.csv", index=False)
