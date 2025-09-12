import streamlit as st
import requests
import pandas as pd
import json
import sys
import io
import requests
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
import os
from openai import OpenAI
from agents import Agent, Runner, WebSearchTool, OpenAIResponsesModel
from openai import AsyncOpenAI
import asyncio

@st.cache_data(show_spinner=False)
def fetch_json(url, params=None, headers=None, timeout=30):
    """Generic cached JSON fetcher."""
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"raw_text": r.text}

@st.cache_data(show_spinner=False)
def get_bootstrap_static():
    return fetch_json("https://fantasy.premierleague.com/api/bootstrap-static/")

@st.cache_data(show_spinner=False)
def get_classic_league_standings(league_id):
    return fetch_json(f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/")

@st.cache_data(show_spinner=False)
def get_entry_history(entry_id):
    return fetch_json(f"https://fantasy.premierleague.com/api/entry/{entry_id}/history/")

@st.cache_data(show_spinner=False)
def get_entry_picks(entry_id, gw):
    return fetch_json(f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gw}/picks/")

@st.cache_data(show_spinner=False)
def get_element_summary(player_id):
    return fetch_json(f"https://fantasy.premierleague.com/api/element-summary/{player_id}/")

def run_agent(agent, prompt):
    """Helper to run async agent inside Streamlit"""
    return asyncio.run(Runner.run(agent, prompt))

# ---- Streamlit UI ----
st.set_page_config(page_title="FPL Roundup", layout="wide")
st.title("FPL Roundup")

league_id = st.text_input("Enter Classic League ID", "")
run = st.button("Run")
force_refresh = st.checkbox("Force refresh (ignore cache)", value=False)

if force_refresh:
    fetch_json.clear()
    get_bootstrap_static.clear()
    get_classic_league_standings.clear()
    get_entry_history.clear()
    get_entry_picks.clear()
    get_element_summary.clear()

if "run_pressed" not in st.session_state:
    st.session_state.run_pressed = False

if run and league_id:
    st.session_state.run_pressed = True

if st.session_state.run_pressed and league_id.strip():
    with st.spinner("Running analysis..."):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        BASE_URL = "https://fantasy.premierleague.com/api"

        bootstrap = requests.get(f"{BASE_URL}/bootstrap-static/").json()

        # Map: player_id -> player object for quick lookup
        players = {p["id"]: p for p in bootstrap["elements"]}

        # Map: team_id -> team name
        teams_lookup = {t["id"]: t["name"] for t in bootstrap["teams"]}

        # Determine the current gameweek from events
        current_gw = next(event["id"] for event in bootstrap["events"] if event["is_current"])

        def get_formation(picks):
            formation = {"DEF": 0, "MID": 0, "FWD": 0}

            for pick in picks["picks"]:
                if pick["position"] <= 11:  # starters only - to counter bboost
                    if pick["element_type"] == 2:  # DEF
                        formation["DEF"] += 1
                    elif pick["element_type"] == 3:  # MID
                        formation["MID"] += 1
                    elif pick["element_type"] == 4:  # FWD
                        formation["FWD"] += 1

            return f"{formation['DEF']}-{formation['MID']}-{formation['FWD']}"

        def get_full_league_standings_and_name(league_id: int):
            standings = []
            page = 1
            while True:
                url = f"{BASE_URL}/leagues-classic/{league_id}/standings/?page_standings={page}"
                resp = requests.get(url).json()

                results = resp["standings"]["results"]
                standings.extend(results)

                if resp["standings"]["has_next"]:
                    page += 1
                else:
                    break

            league_name = resp["league"]["name"]

            return standings, league_name

        league, league_name = get_full_league_standings_and_name(league_id)
        entries = [team["entry"] for team in league]
        gw = current_gw

        gameweeks = [week for week in range(1, current_gw +1)]
        actual_all_picks = {gw: [] for gw in gameweeks}

        for entry_id in entries:
            for gameweek in range(1, gw + 1):
                url = f"{BASE_URL}/entry/{entry_id}/event/{gameweek}/picks/"
                picks_data = requests.get(url).json()
                actual_all_picks[gameweek].append(picks_data)

        picks_count = defaultdict(int)
        captain_count = defaultdict(int)

        for team in actual_all_picks[current_gw]:
            active_picks = {p["element"]: p for p in team["picks"]}

            # Determine who was captain
            cap_id = next(p["element"] for p in team["picks"] if p["multiplier"] > 1)
            captain_count[cap_id] += 1

            # Count all picks
            for p in active_picks.values():
                picks_count[p["element"]] += 1

        footballers_data = []
        for p in bootstrap["elements"]:
            footballers_data.append({
                "Footballer ID":p["id"] ,
                "Footballer name": p["web_name"],
                "Total points": p["total_points"],
                "GW points": p["event_points"],
                "Real team name": teams_lookup[p["team"]],
                "Real team ID": p["team"],
                "Price (in Millions £)": p["now_cost"] / 10,
                "Price last GW (in Millions £)": p["cost_change_event"] / 10 + (p["now_cost"] / 10),
                "Price difference (in Millions £)": p["cost_change_event"] / 10,
                "Times chosen in squad": picks_count[p["id"]],
                "Times captained": captain_count[p["id"]]
            })

        df_footballers = pd.DataFrame(footballers_data)

        #Add opponent strength information
        raw_strengths = {
            t["id"]: (t["strength_overall_home"] + t["strength_overall_away"]) / 2
            for t in bootstrap["teams"]
        }

        # Scale strengths to 1–10
        min_s, max_s = min(raw_strengths.values()), max(raw_strengths.values())

        def scale_strength(val, min_s, max_s):
            return round(1 + 9 * (val - min_s) / (max_s - min_s), 1)

        team_strengths_scaled = {
            tid: scale_strength(val, min_s, max_s) for tid, val in raw_strengths.items()
        }

        # Current gameweek (finished or current)
        fixtures = requests.get(f"{BASE_URL}/fixtures/?event={current_gw}").json()

        # Map: team_id -> opponent_id
        opponents = {}
        for f in fixtures:
            home, away = f["team_h"], f["team_a"]
            opponents[home] = away
            opponents[away] = home

        # Add opponent info (scaled)
        def get_opponent_info(team_id):
            opp_id = opponents.get(team_id)
            if opp_id is None:
                return None, None
            return teams_lookup[opp_id], team_strengths_scaled[opp_id]

        df_footballers["Last opponent name"] = df_footballers["Real team ID"].apply(
            lambda tid: get_opponent_info(tid)[0]
        )
        df_footballers["Last opponent strength (1-10)"] = df_footballers["Real team ID"].apply(
            lambda tid: get_opponent_info(tid)[1]
        )

        def get_player_points(player_id, gw=current_gw):
            url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
            data = requests.get(url).json()
            for match in data["history"]:
                if match["round"] == gw:
                    return match["total_points"]
            return 0

        def get_player_minutes(player_id, gw=current_gw):
            url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
            data = requests.get(url).json()
            for match in data["history"]:
                if match["round"] == gw:
                    return match["minutes"]
            return 0

        teams_data = []
        captaincy_effectiveness_ratio = {}
        all_top_3_contributors_ids = {}
        all_teams_captain_points = {}

        for idx, entry_id in enumerate(entries):
            entry_info = requests.get(f"{BASE_URL}/entry/{entry_id}/").json()
            player_name = f"{entry_info['player_first_name']} {entry_info['player_last_name']}"
            favourite_team = teams_lookup.get(entry_info.get("favourite_team"))

            picks = actual_all_picks[current_gw][idx]
            transfers = requests.get(f"{BASE_URL}/entry/{entry_id}/transfers/").json()

            # Transfers of recent GW
            gw_transfers = [
                (
                    players[t["element_in"]]["web_name"],
                    players[t["element_out"]]["web_name"],
                    players[t["element_in"]]["event_points"] - players[t["element_out"]]["event_points"],
                )
                for t in transfers if t["event"] == gw
            ]

            # Formation
            formation = get_formation(picks)

            # Extract captain / VC / acting captain
            captain_pick = next(p for p in picks["picks"] if p["is_captain"])
            vice_pick = next(p for p in picks["picks"] if p["is_vice_captain"])
            actual_captain_pick = next(p for p in picks["picks"] if p["multiplier"] > 1)

            chosen_captain_id = captain_pick["element"]
            chosen_captain_name = players[chosen_captain_id]["web_name"]
            chosen_captain_with_times_chosen = int(
                df_footballers.loc[df_footballers["Footballer ID"] == chosen_captain_id, "Times captained"].iloc[0]
            )
            captain_tuple = (chosen_captain_name, chosen_captain_with_times_chosen)

            vice_captain_id = vice_pick["element"]
            vice_captain_name = players[vice_captain_id]["web_name"]

            actual_captain_id = actual_captain_pick["element"]
            actual_captain_name = players[actual_captain_id]["web_name"]
            acting_captain_with_id = (actual_captain_name, actual_captain_id)

            actual_captain_points = players[actual_captain_id]["event_points"]
            actual_captain_tuple = (actual_captain_name, actual_captain_points)

            chosen_captain_opponent_difficulty = None
            chosen_captain_opponent_name = None

            for _, row in df_footballers.iterrows():
                if row["Footballer ID"] == chosen_captain_id:
                    chosen_captain_opponent_difficulty = row["Last opponent strength (1-10)"]
                    chosen_captain_opponent_name = row["Last opponent name"]
            
            chosen_captain_opponent_tuple = (actual_captain_name, chosen_captain_opponent_name, chosen_captain_opponent_difficulty)

            # Auto-sub points
            points_by_autosub = sum(
                players[sub["element_in"]]["event_points"]
                for sub in picks.get("automatic_subs", [])
                if sub.get("event") == current_gw
            )

            # VC acting as captain
            points_by_vc_turned_c = players[vice_captain_id]["event_points"] if actual_captain_id == vice_captain_id else 0

            # Build playing XI / bench info
            playing_XI, bench, playing_XI_with_points, bench_with_points = [], [], [], []
            playing_XI_times_chosen = {}

            for p in picks["picks"]:
                pid = p["element"]
                name = players[pid]["web_name"]
                pts = players[pid]["event_points"]

                if p["multiplier"] != 0:
                    playing_XI.append((name, pid))
                    playing_XI_with_points.append((name, pts))
                    times_chosen = int(df_footballers.loc[df_footballers["Footballer ID"] == pid, "Times chosen in squad"].iloc[0])
                    playing_XI_times_chosen[pid] = times_chosen
                else:
                    bench.append((name, pid))
                    bench_with_points.append((name, pts))

            playing_XI_frequency = [(players[pid]["web_name"], freq) for pid, freq in playing_XI_times_chosen.items()]

            # Transfer hits
            transfers_hits_this_gw = picks["entry_history"]["event_transfers_cost"] if picks["entry_history"]["event"] == gw else 0

            # Contributions & captaincy metrics
            contributions = defaultdict(int)
            total_hits = 0
            total_captain_points = 0
            total_best_points = 0
            top_3_contributors, top_3_percentage = [], []

            for gameweek in range(1, gw + 1):
                data = actual_all_picks[gameweek][idx]
                if "picks" not in data:
                    continue

                total_hits += data["entry_history"]["event_transfers_cost"]

                all_players_gw_points = []
                captain_points = 0

                for p in data["picks"]:
                    player_id = p["element"]
                    multiplier = 1 if p["multiplier"] == 2 else p["multiplier"]  # avoid double reward
                    gw_points = get_player_points(player_id, gameweek) * multiplier
                    contributions[player_id] += gw_points

                    player_points = get_player_points(player_id, gameweek)
                    all_players_gw_points.append(player_points)
                    if p.get("is_captain"):
                        captain_points = player_points

                best_points = max(all_players_gw_points) if all_players_gw_points else 0
                total_captain_points += captain_points
                total_best_points += best_points

                top_3_sorted = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
                top_3_contributors = [(players[pid]["web_name"], pts) for pid, pts in top_3_sorted]

                top_3_percentage = [
                    (players[pid]["web_name"], round(100 * pts / (entry_info["summary_overall_points"] + total_hits), 2))
                    for pid, pts in top_3_sorted
                ]

                all_top_3_contributors_ids[entry_id] = top_3_percentage

            ratio = round(total_captain_points / total_best_points, 3) if total_best_points else 0
            captaincy_effectiveness_ratio[entry_id] = ratio
            all_teams_captain_points[entry_info["name"]] = total_captain_points

            # Append final team data
            teams_data.append({
                "Entry ID": entry_id,
                "Player name": player_name,
                "Team name": entry_info["name"],
                "Favourite team": favourite_team,
                "Total points": entry_info["summary_overall_points"],
                "GW points": entry_info["summary_event_points"],
                "Transfers (In, Out, Points gained)": gw_transfers,
                "Transfer hits this GW": transfers_hits_this_gw,
                "Transfer hits overall": total_hits,
                "Formation": formation,
                "Chosen captain with times captained this GW": captain_tuple,
                "Vice captain": vice_captain_name,
                "Acting captain with points this GW": actual_captain_tuple,
                "Chosen captain with opponent": chosen_captain_opponent_tuple,
                "Acting captain with ID": acting_captain_with_id,
                "Playing XI with ID": playing_XI,
                "Playing XI with times chosen this GW": playing_XI_frequency,
                "Playing XI with points this GW": playing_XI_with_points,
                "Bench with ID": bench,
                "Bench with points this GW": bench_with_points,
                "Chips used": picks.get("active_chip"),
                "Top 3 contributors overall with points": top_3_contributors,
                "Percentage contribution overall": top_3_percentage,
                "Captaincy effectiveness ratio overall": ratio,
                "Points earned by auto-substitutions": points_by_autosub,
                "Additional points earned by VC acting as captain": points_by_vc_turned_c,
            })

        df_teams = pd.DataFrame(teams_data)

        def relative_ranks(overall_ranks_dict):
            # Find max number of gameweeks present
            max_gw = max(len(v) for v in overall_ranks_dict.values())

            # Initialize result dict with empty lists
            result = {entry_id: [] for entry_id in overall_ranks_dict}

            # Process each GW
            for gw in range(max_gw):
                # Collect (entry_id, overall_rank) for this GW
                gw_ranks = {
                    entry_id: ranks[gw]
                    for entry_id, ranks in overall_ranks_dict.items()
                    if gw < len(ranks)  # some teams might have fewer entries
                }
    
                # Sort by overall rank (lower is better)
                sorted_entries = sorted(gw_ranks.items(), key=lambda x: x[1])
    
                # Assign relative ranks (1,2,3,…)
                for rel_rank, (entry_id, _) in enumerate(sorted_entries, start=1):
                    result[entry_id].append(rel_rank)

            return result

        rank_history = {}
        for entry_id in entries:
            url = f"{BASE_URL}/entry/{entry_id}/history/"
            resp = requests.get(url).json()
            history = resp.get("current", [])
            rank_history[entry_id] = [gw["overall_rank"] for gw in history]

        relative_ranked_teams = relative_ranks(rank_history)

        df_rank_history = pd.DataFrame(list(relative_ranked_teams.items()), columns=["Entry ID", "Rankings history"])

        # Merge with your main df
        df_teams = df_teams.merge(df_rank_history, on="Entry ID", how="left")

        diff_col = df_teams["Total points"].diff().fillna(0).astype(int)

        # Insert diff column *after* "value"
        col_position = df_teams.columns.get_loc("Total points") + 1
        df_teams.insert(col_position, "Adjacent points difference", diff_col)

        #Add rank change column
        def rank_change(history):
            if len(history) < 2:
                return "0 ➖"  # not enough history
            
            prev, curr = history[-2], history[-1]
            diff = prev - curr   # positive if improved, negative if worsened

            if diff > 0:   # rank improved
                return f"+{diff} 🟢⬆️"
            elif diff < 0: # rank worsened
                return f"{diff} 🔴⬇️"
            else:
                return "0 ➖"

        df_teams["Rank change"] = df_teams["Rankings history"].apply(rank_change)
        
        #Form Team column that combines team name and player name
        df_teams["Team"] = df_teams["Team name"] + "\n" + df_teams["Player name"]

        # Move it to the start
        first_col = df_teams.pop("Team")
        df_teams.insert(1, "Team", first_col)

        #st.dataframe(df_teams.astype(str))

        live = requests.get(f"https://fantasy.premierleague.com/api/event/{current_gw}/live/").json()
        points = [p["stats"]["total_points"] for p in live["elements"] if p["stats"]["minutes"] > 0]
        avg_points = sum(points) / len(points)

        # Utility: find indices of max/min values with optional absolute value
        def all_extremes(series, metric="max", use_abs=False):
            values = series.abs() if use_abs else series

            if metric == "max":
                extreme_value = values.max()
            elif metric == "min":
                extreme_value = values.min()
            else:
                raise ValueError("metric must be 'max' or 'min'")

            return series[values == extreme_value].index.tolist()

        def names_from_indices(df, indices, column="Team name"):
            return [df.iloc[i][column] for i in indices if i < len(df)]

        def value_from_first_index(df, indices, column):
            if indices:
                return df.iloc[indices[0]][column]
            return None

        rule_based_metrics = []

        # 1 League topper
        idx_league_topper = all_extremes(df_teams["Total points"], "max")
        all_league_toppers = names_from_indices(df_teams, idx_league_topper, "Team name")
        rule_based_metrics.append(f"{all_league_toppers} topped the league")

        # 2 League bottom
        idx_league_bottom = all_extremes(df_teams["Total points"], "min")
        all_league_bottoms = names_from_indices(df_teams, idx_league_bottom, "Team name")
        rule_based_metrics.append(f"{all_league_bottoms} are at the bottom of the league")

        # 4 Max change in GW points (absolute)
        idx_max_change = all_extremes(df_teams["GW points"], "max", use_abs=True)
        all_teams_max_change = names_from_indices(df_teams, idx_max_change, "Team name")
        value_max_change = value_from_first_index(df_teams, idx_max_change, "GW points")

        if value_max_change is not None:
            for idx in idx_max_change:
                total_points = int(df_teams.iloc[idx]["Total points"])
                prev_points = total_points - int(df_teams.iloc[idx]["GW points"])
                max_change_in_points = (
                    f"{all_teams_max_change} showed the maximum change of points over the last week. "
                    f"Their change: {value_max_change} points from {prev_points} to {total_points}."
                )
                rule_based_metrics.append(max_change_in_points)

        # 5 Most points gained
        idx_most_points_gained = all_extremes(df_teams["GW points"], "max")
        all_teams_most_points_gained = names_from_indices(df_teams, idx_most_points_gained, "Team name")
        value_most_points_gained = value_from_first_index(df_teams, idx_most_points_gained, "GW points")
        if value_most_points_gained is not None:
            rule_based_metrics.append(
                f"{all_teams_most_points_gained} gained the most points ({value_most_points_gained}) this Gameweek."
            )

        # 6 Least points gained
        idx_least_points_gained = all_extremes(df_teams["GW points"], "min")
        all_teams_least_points_gained = names_from_indices(df_teams, idx_least_points_gained, "Team name")
        value_least_points_gained = value_from_first_index(df_teams, idx_least_points_gained, "GW points")
        if value_least_points_gained is not None:
            rule_based_metrics.append(
                f"{all_teams_least_points_gained} gained the least points ({value_least_points_gained}) this Gameweek."
            )

        # 7 Highest scoring footballer
        idx_highest_scoring_footballer = all_extremes(df_footballers["GW points"], "max")
        all_highest_scoring_footballers = names_from_indices(df_footballers, idx_highest_scoring_footballer, "Footballer name")
        value_highest_scoring_footballers = value_from_first_index(df_footballers, idx_highest_scoring_footballer, "GW points")
        if value_highest_scoring_footballers is not None:
            rule_based_metrics.append(
                f"{all_highest_scoring_footballers} gained the most points ({value_highest_scoring_footballers}) this Gameweek."
            )

        # 8 Lowest scoring footballer
        idx_lowest_scoring_footballer = all_extremes(df_footballers["GW points"], "min")
        all_lowest_scoring_footballers = names_from_indices(df_footballers, idx_lowest_scoring_footballer, "Footballer name")
        value_lowest_scoring_footballers = value_from_first_index(df_footballers, idx_lowest_scoring_footballer, "GW points")
        if value_lowest_scoring_footballers is not None:
            rule_based_metrics.append(
                f"{all_lowest_scoring_footballers} gained the least points ({value_lowest_scoring_footballers}) this Gameweek."
            )

        # 9 Highest scoring real team
        team_scores = {
            team_id: df_footballers.loc[df_footballers["Real team ID"] == team_id, "GW points"].sum()
            for team_id in range(1, 21)
        }
        if team_scores:
            max_score = max(team_scores.values())
            max_teams = [tid for tid, score in team_scores.items() if score == max_score]
            highest_scoring_real_teams_list = [teams_lookup[tid] for tid in max_teams]
            rule_based_metrics.append(f"{highest_scoring_real_teams_list} scored the most points ({max_score}).")

        # 10 Lowest scoring real team
        if team_scores:
            min_score = min(team_scores.values())
            min_teams = [tid for tid, score in team_scores.items() if score == min_score]
            lowest_scoring_real_teams_list = [teams_lookup[tid] for tid in min_teams]
            rule_based_metrics.append(f"{lowest_scoring_real_teams_list} scored the least points ({min_score}).")

        # 11-12 Best/worst transfers
        max_value, min_value = float("-inf"), float("inf")
        max_transfers, min_transfers = [], []

        for idx, row in df_teams.iterrows():
            for tup in row.get("Transfers (In, Out, Points gained)", []):
                x = tup[2]
                if x > max_value:
                    max_value, max_transfers = x, [(idx, tup)]
                elif x == max_value:
                    max_transfers.append((idx, tup))
                if x < min_value:
                    min_value, min_transfers = x, [(idx, tup)]
                elif x == min_value:
                    min_transfers.append((idx, tup))

        for idx, tup in max_transfers:
            team_name = df_teams.iloc[idx]["Team name"]
            rule_based_metrics.append(
                f"{team_name} got {tup[0]} in and removed {tup[1]} smartly and saw a change of {max_value} points."
            )

        for idx, tup in min_transfers:
            team_name = df_teams.iloc[idx]["Team name"]
            rule_based_metrics.append(
                f"{team_name} got {tup[0]} in and removed {tup[1]} unwisely and saw a change of {min_value} points."
            )

        # 13 Highest price increase
        idx_highest_increased_price = all_extremes(df_footballers["Price difference (in Millions £)"], "max")
        all_highest_price_increase_footballers = [
            df_footballers.iloc[i]["Footballer name"]
            for i in idx_highest_increased_price
            if df_footballers.iloc[i]["Price difference (in Millions £)"] > 0
        ]
        value_highest_increased_price = value_from_first_index(df_footballers, idx_highest_increased_price, "Price difference (in Millions £)")
        if all_highest_price_increase_footballers and value_highest_increased_price is not None:
            rule_based_metrics.append(
                f"{all_highest_price_increase_footballers} had the highest price increase (£{value_highest_increased_price}M) this Gameweek."
            )

        # 14 Highest priced players vs points
        most_expensive_players = df_footballers.sort_values("Price (in Millions £)", ascending=False).head(5)
        price_vs_points = "Highest priced footballers \n"
        for _, row in most_expensive_players.iterrows():
            if get_player_minutes(row["Footballer ID"]) == 0:
                continue
            price_vs_points += f"{row['Footballer name']} valued at {row['Price (in Millions £)']} scored {row['GW points']} points. \n"
        rule_based_metrics.append(price_vs_points.strip())

        # 15 Most chosen players vs points
        most_chosen_players = df_footballers.sort_values("Times chosen in squad", ascending=False).head(5)
        chosen_vs_points = "Most picked footballers \n"
        for _, row in most_chosen_players.iterrows():
            if get_player_minutes(row["Footballer ID"]) == 0:
                continue
            chosen_vs_points += f"{row['Footballer name']} chosen {row['Times chosen in squad']} times scored {row['GW points']} points. \n"
        rule_based_metrics.append(chosen_vs_points.strip())

        # 16 Most captained players vs points
        most_captained_players = df_footballers.sort_values("Times captained", ascending=False).head(5)
        captained_vs_points = "Most times picked as captains \n"
        for _, row in most_captained_players.iterrows():
            if get_player_minutes(row["Footballer ID"]) == 0:
                continue
            captained_vs_points += f"{row['Footballer name']} chosen captain {row['Times captained']} times scored {row['GW points']} points. \n"
        rule_based_metrics.append(captained_vs_points.strip())

        # 18 Chips usage
        chips_usage = []
        for _, row in df_teams.iterrows():
            if row.get("Chips used"):
                chips_usage.append(f"{row['Team name']} used the chip(s): {row['Chips used']}")
        if chips_usage:
            rule_based_metrics.append(" ".join(chips_usage))

        # 19 Rare players (least selected in starting XI across league)
        all_playing_times_played = {}

        # Collect all players that actually played (multiplier > 0)
        for pick in actual_all_picks[current_gw]:
            for player in pick.get("picks", []):
                if player.get("multiplier", 0) != 0:
                    row = df_footballers.loc[df_footballers["Footballer ID"] == player["element"], "Times chosen in squad"]
                    if not row.empty:
                        all_playing_times_played[player["element"]] = int(row.squeeze())

        if all_playing_times_played:
            min_selected = min(all_playing_times_played.values())
            # Map: player_id -> player object for quick lookup
            least_selected_players = {
                pid: count for pid, count in all_playing_times_played.items() if count == min_selected
            }

            # Build {player name: score}
            least_selected_players_vs_scores = {}
            for pid in least_selected_players:
                name = players[pid]["web_name"]
                score = players[pid].get("event_points", 0)
                least_selected_players_vs_scores[name] = score

            rare_players_score = (
                f"Here is/are the least selected player(s) who started {min_selected} times "
                f"and their score(s): {least_selected_players_vs_scores}."
            )
            rule_based_metrics.append(rare_players_score)
        else:
            rule_based_metrics.append("No rare players found this Gameweek.")

        #20 top scoring rare captains
        all_playing_XI_players_ids = []
        all_playing_XI_players_frequencies = []

        for id, freq in all_playing_times_played.items():
            all_playing_XI_players_ids.append(id)
            all_playing_XI_players_frequencies.append(freq)

        who_played_me = defaultdict(list)
        for id in all_playing_XI_players_ids:
            for _, team in df_teams.iterrows():
                p11 = team["Playing XI with ID"]
                for name, pid in p11:
                    if pid == id:
                        who_played_me[id].append(team["Team name"])

        who_captained_me = defaultdict(list)
        for id in all_playing_XI_players_ids:
            for _, team in df_teams.iterrows():
                name, pid = team["Acting captain with ID"]
                if pid == id:
                    who_captained_me[id].append(team["Team name"])
            if len(who_captained_me[id]) == 0:
                who_captained_me[id].append("None")

        all_playing_XI_players_df = pd.DataFrame(who_played_me.items(), columns= ["Footballer ID", "Played in teams"])
        all_playing_XI_players_df = df_footballers.merge(all_playing_XI_players_df, on="Footballer ID")
        captains_info = pd.DataFrame(who_captained_me.items(), columns= ["Footballer ID", "Captained by"])
        all_playing_XI_players_df = all_playing_XI_players_df.merge(captains_info, on="Footballer ID")

        non_zero_captains = df_footballers[df_footballers["Times captained"] > 0]
        rare_captains_cutoff = int(non_zero_captains["Times captained"].nsmallest(1).iloc[-1])
        top_scoring_rare_captains_df = all_playing_XI_players_df[all_playing_XI_players_df["Times captained"] == rare_captains_cutoff].sort_values("GW points", ascending=False)
        top_captain_cutoff = int(top_scoring_rare_captains_df["GW points"].nlargest(1).iloc[-1])
        top_scoring_rare_captains_df = top_scoring_rare_captains_df[top_scoring_rare_captains_df["GW points"] == top_captain_cutoff][["Footballer name", "GW points", "Times captained", "Captained by", "Last opponent name", "Last opponent strength (1-10)"]]
        top_scoring_rare_captains_str = top_scoring_rare_captains_df.to_string(index=False)

        rule_based_metrics.append(f"Here are the top scoring player(s) only chosen captain {rare_captains_cutoff} times. The table consists which team(s) chose them: {top_scoring_rare_captains_str}")

        #st.text(rule_based_metrics)

        rule_based_metrics_text = "\n".join(f"- {metric}" for metric in rule_based_metrics)
        df_teams_text = df_teams.to_string(index=False)

        top_3_df = df_teams.sort_values("Total points", ascending=False).head(3)[["Team", "Total points"]]
        top_3_str = top_3_df.to_string(index=False)
        bottom_3_df = df_teams.sort_values("Total points", ascending=False).tail(3)[["Team", "Total points"]]
        bottom_3_str = bottom_3_df.to_string(index=False)

        all_teams_display_ranking = df_teams[["Team name", "Player name", "GW points", "Total points", "Rank change"]]
        all_teams_display_ranking.insert(0, "Rank", range(1, len(all_teams_display_ranking) + 1))
        all_teams_display_ranking_str = all_teams_display_ranking.to_string(index=False)

        # Best and worst transfer-making teams (net score)
        team_transfer_scores = {}

        for idx, row in df_teams.iterrows():
            transfers = row.get("Transfers (In, Out, Points gained)", [])
            total_score = sum(tup[2] for tup in transfers) if transfers else 0
            team_transfer_scores[idx] = total_score

        best_transfer_str, worst_transfer_str = "", ""

        if team_transfer_scores:  # safety check
            max_score = max(team_transfer_scores.values())
            min_score = min(team_transfer_scores.values())

            best_teams = [df_teams.iloc[idx]["Team name"] for idx, score in team_transfer_scores.items() if score == max_score]
            worst_teams = [df_teams.iloc[idx]["Team name"] for idx, score in team_transfer_scores.items() if score == min_score]

            # Create the required strings
            best_transfer_str = f"{', '.join(best_teams)} : {max_score}"
            worst_transfer_str = f"{', '.join(worst_teams)} : {min_score}"

        league_name_text = f"League name: {league_name}"
        gw_text = f"Gameweek Number: {gw}"

        fallers = []
        risers = []

        for _, row in df_teams.iterrows():
            team = row["Team name"]
            ranks = row["Rankings history"]
            current_rank = ranks[-1]

            # --- FALLER check ---
            if len(ranks) > 1:
                best_rank = min(ranks[:-1])
                gw_best = ranks.index(best_rank) + 1
                fall = current_rank - best_rank
                if fall > 0:
                    fallers.append({
                        "team": team,
                        "change": fall,
                        "from_rank": best_rank,
                        "from_gw": gw_best,
                        "to_rank": current_rank,
                        "to_gw": current_gw
                    })

            # --- RISER check ---
            if len(ranks) > 1:
                worst_rank = max(ranks[:-1])
                gw_worst = ranks.index(worst_rank) + 1
                rise = worst_rank - current_rank
                if rise > 0:
                    risers.append({
                        "team": team,
                        "change": rise,
                        "from_rank": worst_rank,
                        "from_gw": gw_worst,
                        "to_rank": current_rank,
                        "to_gw": current_gw
                    })

        # --- Handle ties ---
        fall_strs, rise_strs = [], []

        if fallers:
            max_fall_change = max(f["change"] for f in fallers)
            max_fallers = [f for f in fallers if f["change"] == max_fall_change]

            for f in max_fallers:
                fall_strs.append(
                    f'Team {f["team"]} fell {f["change"]} places '
                    f'from GW{f["from_gw"]} (rank: {f["from_rank"]}) '
                    f'to GW{f["to_gw"]} (rank: {f["to_rank"]}).'
                )
        else:
            fall_strs = "No rank fallers so far."

        if risers:
            max_rise_change = max(r["change"] for r in risers)
            max_risers = [r for r in risers if r["change"] == max_rise_change]

            for r in max_risers:
                rise_strs.append(
                    f'Team {r["team"]} rose {r["change"]} places '
                    f'from GW{r["from_gw"]} (rank: {r["from_rank"]}) '
                    f'to GW{r["to_gw"]} (rank: {r["to_rank"]}).'
                )
        else: rise_strs = "No rank risers so far."

        least_selected_players_indices = []
        for p in least_selected_players:
            least_selected_players_indices.append(p)

        names = list(least_selected_players_vs_scores.keys())
        points = list(least_selected_players_vs_scores.values())

        # Step 2: build DataFrame
        df_rare_players = pd.DataFrame({
            "Player ID": least_selected_players_indices,
            "Player": names,
            "Score": points
        })

        df_rare_players = df_rare_players.merge(
            df_footballers[["Footballer ID", "Last opponent name", "Last opponent strength (1-10)"]],
            left_on="Player ID",
            right_on="Footballer ID",
            how="left"
        )

        # Drop the redundant "Footballer ID" column after merge
        df_rare_players = df_rare_players.drop(columns=["Footballer ID"])

        df_rare_players = df_rare_players.sort_values("Score", ascending=False).head(3)

        top_rare_indices = list(df_rare_players.head(3)["Player ID"])
        selected_by = defaultdict(list)
        for index in top_rare_indices:
            for _, row in df_teams.iterrows():
                team = row["Playing XI with ID"]
                for player in team:
                    if player[1] == index:
                        selected_by[index].append(row["Team name"])

        rare_player_team = list(selected_by.values())

        df_rare_players_shortlisted = df_rare_players.copy().drop(columns = "Player ID")
        df_rare_players_shortlisted["Selected by"] = rare_player_team
        top_unique_picks = df_rare_players_shortlisted.to_string(index=False)

        df_ratios = pd.DataFrame(list(captaincy_effectiveness_ratio.items()), columns=["Entry ID", "Ratio"])

        # Merge with df_team to get team names
        df_ratios = df_ratios.merge(df_teams[["Entry ID", "Team name"]], on="Entry ID", how="right")

        # Sort by ratio
        df_ratios_sorted = df_ratios.sort_values(by="Ratio", ascending=False)

        top_cutoff = df_ratios_sorted["Ratio"].nlargest(3).iloc[-1]
        top_captaincy_ratio_teams = df_ratios_sorted[df_ratios_sorted["Ratio"] >= top_cutoff].reset_index(drop=True)
        top_captaincy_ratio_teams = top_captaincy_ratio_teams.drop(columns="Entry ID")
        top_captaincy_str = top_captaincy_ratio_teams.to_string(index=False)

        bottom_cutoff = df_ratios_sorted["Ratio"].nsmallest(3).iloc[-1]
        bottom_captaincy_ratio_teams = df_ratios_sorted[df_ratios_sorted["Ratio"] <= bottom_cutoff].reset_index(drop=True)
        bottom_captaincy_ratio_teams = bottom_captaincy_ratio_teams.drop(columns="Entry ID")
        worst_captaincy_str = bottom_captaincy_ratio_teams.to_string(index=False)

        all_top_1_contribution_records = []
        for entry_id, tuples in all_top_3_contributors_ids.items():
            team_name = df_teams[df_teams["Entry ID"] == entry_id]["Team name"].iloc[0]
            for player, score in tuples:
                all_top_1_contribution_records.append((team_name, player, score))
                break

        # Convert to DataFrame
        df_contribution_records = pd.DataFrame(all_top_1_contribution_records, columns=["Team name", "Highest reliance player name", "Reliance %"])

        # Find the cutoff for top 3 scores (handles ties)
        top_3_reliance_cutoff = df_contribution_records["Reliance %"].nlargest(3).iloc[-1]
        bottom_3_reliance_cutoff = df_contribution_records["Reliance %"].nsmallest(3).iloc[-1]

        top_3_reliance_df = df_contribution_records[df_contribution_records["Reliance %"] >= top_3_reliance_cutoff].sort_values(by="Reliance %", ascending=False)
        top_3_reliance_str = top_3_reliance_df.to_string(index=False)
        bottom_3_reliance_df = df_contribution_records[df_contribution_records["Reliance %"] <= bottom_3_reliance_cutoff].sort_values(by="Reliance %", ascending=True)
        bottom_3_reliance_str = bottom_3_reliance_df.to_string(index=False)

        # Aggregate chip effectiveness per entry (sum of chip impacts)
        chips_scores = {entry_id: 0 for entry_id in entries}
        overall_chips_used = {}

        for idx, entry_id in enumerate(entries):
            total_chips = []
            for gw in range(1, current_gw + 1):
                picks = actual_all_picks[gw][idx]

                if "picks" not in picks:  # skip if data not available
                    continue
    
                chip_used = picks.get("active_chip")
                total_chips.append(chip_used) # Only 1 chip can be used per gameweek in FPL
                if not chip_used:
                    continue

                if chip_used == "3xc": # Triple captain
                    captain_id = next(p["element"] for p in picks["picks"] if p["multiplier"] > 1)
                    added_points = players[captain_id]["event_points"]
                    chips_scores[entry_id] += added_points

                elif chip_used == "bboost": # Bench boost
                    bench = [p["element"] for p in picks["picks"] if p["position"] >11]
                    bench_points = sum(players[eid]["event_points"] for eid in bench)
                    chips_scores[entry_id] += bench_points

                elif chip_used == "freehit":
                    # Actual FH score
                    actual_points = sum(
                        players[p["element"]]["event_points"] * p["multiplier"]
                        for p in picks["picks"]
                    )

                    prev_picks = actual_all_picks.get(gw-1, {})[idx]

                    if "picks" in prev_picks:
                        hypothetical_points = sum(
                            players[p["element"]]["event_points"] * p["multiplier"]
                            for p in prev_picks["picks"]
                        )
                        added_points = actual_points - hypothetical_points
                        chips_scores[entry_id] += added_points

                elif chip_used == "wildcard":
                    # Actual WC score
                    actual_points = sum(
                        players[p["element"]]["event_points"] * p["multiplier"]
                        for p in picks["picks"]
                    )

                    prev_url = f"{BASE_URL}/entry/{entry_id}/event/{gw-1}/picks/"
                    prev_picks = requests.get(prev_url).json()

                    if "picks" in prev_picks:
                        hypothetical_points = sum(
                            players[p["element"]]["event_points"] * p["multiplier"]
                            for p in prev_picks["picks"]
                        )
                        added_points = actual_points - hypothetical_points
                        chips_scores[entry_id] += added_points
            overall_chips_used[entry_id] = total_chips

        # Convert results to DataFrame with team names
        results = []
        for entry_id, score in chips_scores.items():
            results.append({"Entry ID": entry_id, "Season chips score": score, "Overall chips used":overall_chips_used[entry_id]})

        df_chips = pd.DataFrame(results)
        df_chips = df_chips.merge(df_teams, on="Entry ID")
        df_chips = df_chips[["Team name", "Player name", "Overall chips used", "Season chips score"]]

        top_3_chips_cutoff = df_chips["Season chips score"].nlargest(3).iloc[-1]
        bottom_3_chips_cutoff = df_chips["Season chips score"].nsmallest(3).iloc[-1]

        top_3_chips_df = df_chips[df_chips["Season chips score"] >= top_3_chips_cutoff].sort_values(by="Season chips score", ascending=False)
        top_3_chips_str = top_3_chips_df.to_string(index=False)
        bottom_3_chips_df = df_chips[df_chips["Season chips score"] <= bottom_3_chips_cutoff].sort_values(by="Season chips score", ascending=True)
        bottom_3_chips_str = bottom_3_chips_df.to_string(index=False)

        all_teams_captain_points_df = pd.DataFrame(all_teams_captain_points.items(), columns = ["Entry ID", "Total captain points overall"])
        top_3_highest_scoring_captains_cutoff = all_teams_captain_points_df["Total captain points overall"].nlargest(3).iloc[-1]
        top_3_highest_scoring_captains_df = all_teams_captain_points_df[all_teams_captain_points_df["Total captain points overall"] >= top_3_highest_scoring_captains_cutoff].sort_values("Total captain points overall", ascending=False)
        top_3_highest_scoring_captains_str = top_3_highest_scoring_captains_df.to_string(index=False)

        bottom_3_lowest_scoring_captains_cutoff = all_teams_captain_points_df["Total captain points overall"].nsmallest(3).iloc[-1]
        bottom_3_lowest_scoring_captains_df = all_teams_captain_points_df[all_teams_captain_points_df["Total captain points overall"] <= bottom_3_lowest_scoring_captains_cutoff].sort_values("Total captain points overall", ascending=False)
        bottom_3_lowest_scoring_captains_str = bottom_3_lowest_scoring_captains_df.to_string(index=False)

        automatic_substitutions_df = df_teams[["Team name", "Player name", "Points earned by auto-substitutions"]]
        top_3_highest_scoring_autosubs_cutoff = automatic_substitutions_df["Points earned by auto-substitutions"].nlargest(3).iloc[-1]
        top_3_highest_scoring_autosubs_df = automatic_substitutions_df[automatic_substitutions_df["Points earned by auto-substitutions"] >= top_3_highest_scoring_autosubs_cutoff].sort_values("Points earned by auto-substitutions", ascending=False)
        top_3_highest_scoring_autosubs_str = top_3_highest_scoring_autosubs_df.to_string(index=False)

        bottom_3_lowest_scoring_autosubs_cutoff = automatic_substitutions_df["Points earned by auto-substitutions"].nsmallest(3).iloc[-1]
        bottom_3_lowest_scoring_autosubs_df = automatic_substitutions_df[automatic_substitutions_df["Points earned by auto-substitutions"] <= bottom_3_lowest_scoring_autosubs_cutoff].sort_values("Points earned by auto-substitutions", ascending=False)
        bottom_3_lowest_scoring_autosubs_str = bottom_3_lowest_scoring_autosubs_df.to_string(index=False)

        vc_turned_c_df = df_teams[["Team name", "Player name", "Additional points earned by VC acting as captain"]]
        top_3_highest_scoring_vc2c_cutoff = vc_turned_c_df["Additional points earned by VC acting as captain"].nlargest(3).iloc[-1]
        top_3_highest_scoring_vc2c_df = vc_turned_c_df[vc_turned_c_df["Additional points earned by VC acting as captain"] >= top_3_highest_scoring_vc2c_cutoff].sort_values("Additional points earned by VC acting as captain", ascending=False)
        top_3_highest_scoring_vc2c_str = top_3_highest_scoring_vc2c_df.to_string(index=False)

        bottom_3_lowest_scoring_vc2c_cutoff = vc_turned_c_df["Additional points earned by VC acting as captain"].nsmallest(3).iloc[-1]
        bottom_3_lowest_scoring_vc2c_df = vc_turned_c_df[vc_turned_c_df["Additional points earned by VC acting as captain"] <= bottom_3_lowest_scoring_vc2c_cutoff].sort_values("Additional points earned by VC acting as captain", ascending=False)
        bottom_3_lowest_scoring_vc2c_str = bottom_3_lowest_scoring_vc2c_df.to_string(index=False)

        df_teams_times_captained_strip = df_teams[["Team name", "Player name", "Chosen captain with times captained this GW", "Acting captain with points this GW"]]
        df_teams_times_captained_strip_str = df_teams_times_captained_strip.to_string(index=False)

        df_teams_times_chosen_strip = df_teams[["Team name", "Player name", "Playing XI with times chosen this GW"]]
        df_teams_times_chosen_strip_str = df_teams_times_chosen_strip.to_string(index=False)

        chip_usage_stats = df_teams[["Team name", "Player name", "Chips used"]]
        chip_usage_stats_str = chip_usage_stats.to_string(index=False)

        favourite_teams_df = df_teams[["Team name", "Player name", "Favourite team"]]
        favourite_teams_str = favourite_teams_df.to_string(index=False)

        standings_df = df_teams[["Team name", "Player name", "Total points"]]
        standings_str = standings_df.to_string(index=False)

        # build table with winner/loser
        fixtures = requests.get(f"https://fantasy.premierleague.com/api/fixtures/?event={current_gw}" ).json()
        matches = []
        for f in fixtures:
            if f["finished"]:
                home_team = teams_lookup[f["team_h"]]
                away_team = teams_lookup[f["team_a"]]
                home_score = f["team_h_score"]
                away_score = f["team_a_score"]

                # decide result
                if home_score > away_score:
                    winner, loser = home_team, away_team
                elif away_score > home_score:
                    winner, loser = away_team, home_team
                else:
                    winner, loser = "Draw", "Draw"

                matches.append({
                    "GW": f["event"],
                    "Home": home_team,
                    "Away": away_team,
                    "Score": f"{home_score} - {away_score}",
                    "Winner": winner,
                    "Loser": loser
                })

        real_gameweek_results_df = pd.DataFrame(matches)
        real_gameweek_results_str = real_gameweek_results_df.to_string(index=False)

        real_gameweek_results_str = f"Here are the gameweek {current_gw} results: \n"
        for i, result in real_gameweek_results_df.iterrows():
            if result["Winner"] == "Draw":
                real_gameweek_results_str += f"{result["Home"]} drew {result["Away"]} with a {result["Score"]} score. \n"
            else:
                if result["Home"] == result["Winner"]:
                    real_gameweek_results_str += f"{result["Winner"]} defeated {result["Loser"]} with a {result["Score"]} score at home. \n"
                else:
                    real_gameweek_results_str += f"{result["Winner"]} defeated {result["Loser"]} with a {result["Score"]} score at away. \n"

        df_teams_stripped = df_teams.copy()
        df_teams_stripped = df_teams_stripped.drop(columns=["Team name", "Player name", "Entry ID", "Playing XI with ID", "Adjacent points difference", "Transfer hits this GW", "Transfer hits overall", "Bench with ID", "Formation"])
        df_teams_stripped_str = df_teams_stripped.to_string(index=False)

        min_pd = float("-inf")
        min_row = 0
        for index, row in df_teams.head(5).iterrows():
            if index == 0:
                continue
            if row["Adjacent points difference"] > min_pd:
                min_pd = row["Adjacent points difference"]
                min_row = index

        rival_1 = df_teams.drop(columns=["Entry ID", "Adjacent points difference"]).iloc[min_row - 1]
        rival_1_str = rival_1.to_string(index=False)
        rival_2 = df_teams.drop(columns=["Entry ID", "Adjacent points difference"]).iloc[min_row]
        rival_2_str = rival_2.to_string(index=False)

        openai_agent_client = AsyncOpenAI(api_key=api_key)

        fpl_agent_links = Agent(
            name="FPLScoutAgent",
            instructions=(
            "You are an assistant that fetches official Fantasy Premier League (FPL) Scout articles "
            "for the given gameweek of the 2025/26 Premier League season. "
            "STRICT RULES you must follow:\n"
            "1. Only use articles from the official Premier League website (premierleague.com).\n"
            "2. The article MUST have a publishing date of August 2025 or later. If the date is missing or earlier, ignore it.\n"
            "3. The article MUST explicitly be about the 2025/26 season. If the season is not clearly stated, skip it.\n"
            "4. The article MUST correspond to the requested gameweek number. Ignore GW1 articles or 'opening week' articles when asked for GW2, etc.\n"
            "5. If you are not absolutely sure that the article matches all the above, skip it.\n"
            "6. Return only a numbered list of manager tips/advice/suggestions from valid articles. No links. No explanations.\n"
            "7. NEVER include articles from previous seasons (2024/25, 2023/24, etc.), even if they look similar."
        ),
            tools=[WebSearchTool()],
            model=OpenAIResponsesModel(
                "gpt-4o",
                openai_client=openai_agent_client   # 👈 important!
            ),
        )

        result = run_agent(
            fpl_agent_links,
            f"""Extract only tips/advice/suggestions from official FPL Scout articles for Gameweek {current_gw} of the 2025/26 season. 
        Rules you must follow strictly:
        - Only include articles published August 2025 or later.
        - Only include articles that explicitly say they are for Gameweek {current_gw}.
        - Skip any article from previous seasons (2024/25 or earlier).
        - If unsure about the season or gameweek → do NOT include it.
        Return only a numbered list of the tips/advice/suggestions. No links, no explanations.
        """
        )

        prompt_common_choices = f"""Talk about the most common strategy throughout all managers in our league. 
        Did our managers play safe by playing most commonly chosen footballers and captaining commonly captained players? Or did they take any risks? Did they captain players who were playing against easy opponents?
        Here is the data for the captains of our league's teams: {df_teams_times_captained_strip_str}. 
        It includes the number of times a captain was shared between teams as the third column. Every tuple (A,x) means Captain named A was captained x times in our league this gameweek.
        In the third column, it also includes the opponent name, and their difficulty rating from 1 to 10. It's generally safer to captain players who are having an easy fixture, to yield the most points. (A,B,x) means A is the captain name, B is the team name they played against, and x is that team's difficulty rating from 1 to 10 with 1 being the easiest.
        Try to comment about the opponent difficulty.
        If the acting captain is different than the chosen captain, that captain didn't play the match for some reason. 
        Make sure to mention this in your commentary if a captain didn't play meaning the armband was automatically switched to the Vice-Captain.
        Important FPL rule: FPL rewards double points for the captain by default, so if the data says a captain scored 5 points, those are 5 raw points before doubling, i.e., the team actually got 10 points from him.


        Similarly, here's the data for commonly chosen footballers by our managers: {df_teams_times_chosen_strip_str}. In the last column,
        you'd see the player name along with the number of times managers chose them in their team in this Gameweek. Every tuple (A,x) means Footbalelr named A was chosen x times in the team in our league this gameweek.

        Also here are the chip usage statistics by each team for this week: {chip_usage_stats_str}. Look at them, and comment whether there was any chip
        that was commonly used THIS GAMEWEEK. 
        Include all comments but make sure your response is short (50-100 words)
        When discussing managers, don't take their full names, just the first name is enough"""

        response_common_choices = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty, mean-spirited, banterful and funny fantasy football commissioner."},
                {"role": "user", "content": prompt_common_choices}
            ],
            temperature=0.7
        )

        #print(response_common_choices.choices[0].message.content)

        prompt_rare_choices = f"""Give shoutout to managers who played rarely chosen players and scored well. Use this data: {top_unique_picks}. 
        Also, here are rarely chosen captains and their scores: {top_scoring_rare_captains_str}. 
        It includes how many points the footballer scored, how many times he was picked as the captain, and by whom. Importantly, the last two columns tell which team they were playing against, and the difficulty rating of that team from (1-10), with 1 being the easiest.
        Try to comment about the opponent difficulty.
        Check {chip_usage_stats} to see if anyone stood out with chip usage this week. Limit your response within 50-100 words.
        When discussing managers, don't take their full names, just the first name is enough.
        Important FPL rule:FPL rewards double points for the captain by default, so if the data says a captain scored 5 points, those are 5 raw points before doubling, i.e., the team actually got 10 points from him.
        """
        response_rare_choices = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty, mean-spirited, banterful and funny fantasy football commissioner."},
                {"role": "user", "content": prompt_rare_choices}
            ],
            temperature=0.7
        )

        #print(response_rare_choices.choices[0].message.content)

        prompt_fixture_results = f"""Use the real gameweek results to trash talk about managers' favourite teams. Here are the results: {real_gameweek_results_str}. 
        Here are managers and their favourite teams: {favourite_teams_str}. Only comment if a result is surprising/unexpected, or if a team won by a big margin, or if it's usual to make fun of a team everytime they play bad.
        Keep it short, in 50-100 words, but make sure to sound mean and funny. When discussing managers, don't take their full names, just the first name is enough.
        IMPORTANT: If many managers support the same team and you decide to comment on them, DO NOT take their names individually, just use Team X supporters, DO NOT mention their names.
        """

        response_fixture_results = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty, mean-spirited, banterful and funny fantasy football commissioner."},
                {"role": "user", "content": prompt_fixture_results}
            ],
            temperature=0.7
        )

        #print(response_fixture_results.choices[0].message.content)

        prompt_rivals = f"""
        You are analyzing a rivalry between two close FPL teams:
        1. {rival_1}
        2. {rival_2}

        Tasks:
        - State their rivalry and declare how many points separate them.
        - Give their current rankings (last element in 'Rankings history' list).
        - Suggest concrete what-if scenarios for either team (how one could overtake or extend the lead).

        Strict rules:
        - Do NOT comment on formations.
        - Acting captain = chosen vice-captain. If that’s the case, no what-if should mention swapping captains.
        - Use 'Playing XI with points this GW' for starters and 'Bench with points this GW' for unused players. Do not suggest bench players into XI.
        - Use 'Transfers (In, Out, Points gained)' to highlight wasted transfers (e.g., negative or 0 point gain).
        - Captains earn double points. So if another player had 1 more point, captaining them means +2 overall.
        - Only suggest scenarios you are very certain about.

        Important FPL rule: FPL rewards double points for the captain by default, so if the data says a captain scored 5 points, those are 5 raw points before doubling, i.e., the team actually got 10 points from him.
        Keep response concise (50-100 words).
        """

        response_rivals = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty, mean-spirited, banterful and funny fantasy football commissioner."},
                {"role": "user", "content": prompt_rivals}
            ],
            temperature=0.7
        )

        #print(response_rivals.choices[0].message.content)

        prompt_reliance = f"""Use this data to comment about players who relied the most on a single player for their scores: {top_3_reliance_str}
        Similarly, use this for players who relied the least on a single player for their scores: {bottom_3_reliance_str}. 
        It is usually good if a player does not rely on a single footballer too much. Keep your response short, within 50 words.
        """

        response_reliance = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty and funny fantasy football commissioner."},
                {"role": "user", "content": prompt_reliance}
            ],
            temperature=0.7
        )

        #print(response_reliance.choices[0].message.content)

        prompt_chips = f"""Here are some statistics about the points earned by managers using chips. Top chip users throughout the season: {top_3_chips_str}.
        Worst chip users throughout the season: {bottom_3_chips_str}.
        Comment on how any team has so far earned from the use of chips, you'll also see how many and which chips they have used overall in this season.
        You have a list of chip usage: It is chip usage per week. Eg if the list looks like [bboost, None, None, 3xc], it means the manager used Bench Boost in GW 1, none in GW 2 and 3, and Triple Captain in GW 4. 
        Important: Make sure to mention in WHICH GAMEWEEK the manager used their chips, if any.
        Keep your response within 100 words.
        """

        response_chips = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty, mean-spirited, banterful and funny fantasy football commissioner."},
                {"role": "user", "content": prompt_chips}
            ],
            temperature=0.7
        )

        #print(response_chips.choices[0].message.content)

        #Clean chips df
        def clean_chips(chips_list):
            result = []
            for i, chip in enumerate(chips_list, start=1):  # start=1 so GW = index+1
                if chip is not None:
                    result.append((chip, i))
            return result

        top_3_chips_df["Chips used with gameweek"] = top_3_chips_df["Overall chips used"].apply(clean_chips)
        bottom_3_chips_df["Chips used with gameweek"] = bottom_3_chips_df["Overall chips used"].apply(clean_chips)
        top_3_chips_df = top_3_chips_df.drop(columns=["Overall chips used"])
        bottom_3_chips_df = bottom_3_chips_df.drop(columns=["Overall chips used"])
        top_3_chips_str = top_3_chips_df.to_string(index=False)
        bottom_3_chips_str = bottom_3_chips_df.to_string(index=False)

        prompt_captaincy = f"""Comment on how well someone chose their captain (one should choose their highest scoring player as captain always). A ratio of 1 means the best player chosen as captain. 0 means worst. Check these tables: {top_captaincy_str}, {worst_captaincy_str}.
        Keep your answer within 50-70 words. Instead of using the word ratio, use captaincy effectiveness ratio. Also, try to convert them into words rather than throwing the stats.
        Important FPL rule: FPL rewards double points for the captain by default, so if the data says a captain scored 5 points, those are 5 raw points before doubling, i.e., the team actually got 10 points from him. 
        """

        response_captaincy = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty, mean-spirited, banterful and funny fantasy football commissioner."},
                {"role": "user", "content": prompt_captaincy}
            ],
            temperature=0.7
        )

        #print(response_captaincy.choices[0].message.content)

        prompt_rise_fall = f"""These are the teams with the highest increase in rank to reach to their current rank: {rise_strs}. 
        These are the teams with the highest fall in rank to reach their current rank: {fall_strs}.
        Comment on them in under 50 words. Mention their ranks, how many places they fell/rose as well.
        """

        response_rise_fall = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty, mean-spirited, banterful and funny fantasy football commissioner."},
                {"role": "user", "content": prompt_rise_fall}
            ],
            temperature=0.7
        )
        #print(response_rise_fall.choices[0].message.content)


        prompt_luck = f"""Luck factor: Here are the teams with most points from automatic-substitutions: {top_3_highest_scoring_autosubs_str}. Here are the teams with least points from automatic-substitutions: {bottom_3_lowest_scoring_autosubs_str}.
        Here are the teams with most points from Vice Captain acting as Captain: {top_3_highest_scoring_vc2c_str}. Here are the teams with least points from Vice Captain acting as Captain: {bottom_3_lowest_scoring_vc2c_str}. 
        Use this data to comment on how lucky a team got, and how. Usually, auto-substitution points are counted as bonus. Vice captaining acting captain means their points will be doubled so if a VC scored great for a team, while acting as a captain, they got lucky.
        Keep your response under a 100 words. Only comment on non-zero scores from the teams. A 0 score can mean there was no auto-substition or no captaincy switching requirement.
        """

        response_luck = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty, mean-spirited, banterful and funny fantasy football commissioner."},
                {"role": "user", "content": prompt_luck}
            ],
            temperature=0.7
        )

        #print(response_luck.choices[0].message.content)

        top_team_season_ranking_history = df_teams[["Rankings history", "Team name", "Player name"]].head(1)
        bottom_team_season_ranking_history = df_teams[["Rankings history", "Team name", "Player name"]].tail(1)

        prompt_season_performance_top_bottom = f"""Use rank history to comment on the leader and loser’s season summary and performance.
        Leader's ranking history: {top_team_season_ranking_history}
        Bottom team's ranking history: {bottom_team_season_ranking_history}
        Keep your response under 70 words. Make sure to mention which team is the leader and which is at the bottom.
        """

        response_season_performance_top_bottom = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty, mean-spirited, banterful and funny fantasy football commissioner."},
                {"role": "user", "content": prompt_season_performance_top_bottom}
            ],
            temperature=0.7
        )

        #print(response_season_performance_top_bottom.choices[0].message.content)

        prompt_additional_metrics = f"""Here are some additional metrics we found from our private FPL league, its teams and footballers used:
        {rule_based_metrics_text}
        If you find anything interesting, do comment on it. Keep your response within a 100 words.
        """

        response_additional_metrics = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty, mean-spirited, banterful and funny fantasy football commissioner."},
                {"role": "user", "content": prompt_additional_metrics}
            ],
            temperature=0.7
        )

        #print(response_additional_metrics.choices[0].message.content)

        footballers_scores = df_footballers[["Footballer name", "GW points"]]
        footballers_scores_str = footballers_scores.to_string(index=False)

        df_rare_players_shortlisted_stripped = df_rare_players_shortlisted.drop(columns=["Last opponent name", "Last opponent strength (1-10)"])
        top_unique_picks_stripped = df_rare_players_shortlisted_stripped.to_string(index=False)

        top_scoring_rare_captains_df_stripped = top_scoring_rare_captains_df.drop(columns=["Last opponent name", "Last opponent strength (1-10)"])
        top_scoring_rare_captains_str_stripped = top_scoring_rare_captains_df_stripped.to_string(index=False)

        prompt_final = f""" You are an FPL commissioner commenting on a private FPL league. Make sure to sound witty, mean-spirited, funny, banterful. I need hardcore banter as if it’s a made for a bunch of London lads in their early 20’s. Be as brutal as you can.
        Here are bits and pieces of responses from various LLM calls, commenting on various aspects of our FPL private league. For some pieces, I have provided you with tables,
        you MUST add them after you add that bit of information in your final response. Give response in the markdown format, including the tables.
        Some values in the tables consist of [] brackets, remove these brackets when displaying them in your response.

        - Common player/captain/chips choices: {response_common_choices.choices[0].message.content}
        - Rare player/captain choices: {response_rare_choices.choices[0].message.content}. 
        MUST print these two tables in your output immediately with the above content. 1) Title: Top scoring rarely selected players. Data: {top_unique_picks_stripped}. 2) Title: Top scoring rarely captained players. Data: {top_scoring_rare_captains_str_stripped}.
        - Trash talk on real-team fixtures: {response_fixture_results.choices[0].message.content}
        - Rival managers in the league: {response_rivals.choices[0].message.content}
        - Most/Least single-player reliant teams: {response_reliance.choices[0].message.content}. 
        MUST print these two tables in your output immediately with the above content. 1) Title: Most single-player reliant teams. Data: {top_3_reliance_str}. 2) Title: Least single-player reliant teams. Data: {bottom_3_reliance_str}.
        - Points earned from chips: {response_chips.choices[0].message.content}. 
        MUST print these two tables in your output immediately with the above content. 1) Title: Teams with most points from chips. Data: {top_3_chips_str}. 2) Title: Teams with least points from chips. Data: {bottom_3_chips_str}
        - Captaincy effectiveness of teams: {response_captaincy.choices[0].message.content}
        MUST print these two tables in your output immediately with the above content. 1) Title: Top captaincy effectiveness teams. Data: {top_captaincy_str}. 2) Title: Bottom captaincy effectiveness teams. Data: {worst_captaincy_str}.
        - Teams that rose and fell the most in standings to get to their current positions: {response_rise_fall.choices[0].message.content}
        - Points earned luckily, by automatic substitutions and vice-captain turning captain in their absense: {response_luck.choices[0].message.content}.
        MUST print these FOUR tables in your output immediately with the above content. 1) Title: Most points earned by auto-subs, Data: {top_3_highest_scoring_autosubs_str}. 2)Title: Least points earned by auto-subs. Data: {bottom_3_lowest_scoring_autosubs_str}. 3) Title: Most points earned by Vice Captain acting as Captain. Data: {top_3_highest_scoring_vc2c_str}. 4) Title: Least points earned by Vice Captain acting as Captain. Data: {bottom_3_lowest_scoring_vc2c_str}.
        - Season-wide rank of league leader and bottom placed team: {response_season_performance_top_bottom.choices[0].message.content}
        - More metrics you can add to your commentary. Make sure nothing is repeated or conflicting to what's already mentioned though: {response_additional_metrics.choices[0].message.content}

        Compile them together, make sure there is no redundancy of information. No repetition of information, no self-conflicting statements. You can change the order of information in these responses.
        You should also rephrase and regroup the information provided, but don't change any information.
        Make sure your final response is only 700-1000 words long apart from the tables that you're printing.

        IMPORTANT: Here are some tips that the FPL Scout from the official premier league website gave for this gameweek. 
        It mostly includes adding/removing certain players or using certain chips which the scout thinks will prove to be fruitful.
        Read through these tips: {result.final_output}

        And using your knowledge and data, comment on the following:
        1. If any of these tips would have yielded good points, now that the gameweek has ended and we have the results in our hands.
        2. Did any manager follow any of these tips? And did they get a good repsonse from it?
        3. To check footballers and how many points they scored, you can check this list. Disclaimer: It's quite long: {footballers_scores_str}.

        You can look for information accross these responses and combine them together as well. You can change the tone/phrasing of these responses as well.
        If you mention words like highest scoring, lowest ranked, highest ratio, and other numeric terms, also give the number with the information. 
        Eg. Instead of Footballer A was the highest scoring footballer this week, you should say Footballer A was the highest scoring footballer this week with x points.
        """

        response_final = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a witty fantasy football commissioner."},
                {"role": "user", "content": prompt_final}
            ],
            temperature=0.7
        )

        #print(response_final.choices[0].message.content)

        
        show_internal = st.toggle("Show internal database", value=False)  # default off
    
        if show_internal:
            st.write("Footballers dataframe")
            st.dataframe(df_footballers.iloc[400:450])
            st.write("Teams Dataframe")
            st.dataframe(df_teams.astype(str))
            st.write("All footballers playing for our managers")
            st.dataframe(all_playing_XI_players_df)
            st.write("Rule based metrics list")
            st.write(rule_based_metrics)
            st.text("Web-search result for FPL Scout tips")
            st.text(result.final_output)

        st.markdown("# Final Report")

        st.markdown(f"\n #### {league_name_text}")
        st.markdown(f"\n #### {gw_text}")
        st.markdown(f"### Summary: \n\n""")

        st.markdown(f"{response_final.choices[0].message.content} \n\n")
        st.text(f"\nLeague Table Standings\n")
        all_teams_display_ranking["Team"] = (
            "**" + all_teams_display_ranking["Team name"] + "**"
            + "<br><span style='color:gray; font-size:90%'>"
            + all_teams_display_ranking["Player name"] + "</span>"
        )
        all_teams_display_ranking = all_teams_display_ranking.drop(columns=["Team name", "Player name"])
        st.markdown(all_teams_display_ranking.to_markdown(index=False), unsafe_allow_html=True)
        st.text(f"\nBest Transfer Maker(s): \n{best_transfer_str} points earned in total through transfers\n\n")    
        st.text(f"\nWorst Transfer Maker(s): \n{worst_transfer_str} points earned in total through transfers\n\n")
        st.text(f"\nBiggest rank riser(s): \n{", ".join(rise_strs)} \n\n")
        st.text(f"\nBiggest rank faller(s): \n{", ".join(fall_strs)} \n\n")
        st.text(f"\nTeams with the highest scoring captains:\n")
        st.table(top_3_highest_scoring_captains_df.reset_index(drop=True))
        st.text(f"\nTeams with the lowest scoring captains:\n")
        st.table(bottom_3_lowest_scoring_captains_df.reset_index(drop=True))

elif run and not league_id.strip():
    st.warning("Please enter a valid League ID.")
