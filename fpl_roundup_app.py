import streamlit as st
import requests
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_URL = "https://fantasy.premierleague.com/api"

st.title("FPL Roundup")

league_id = st.text_input("Enter League ID")

#Take league ID input
if st.button("Run") and league_id:
    st.write("Processing League ID:", league_id)
    
    bootstrap = requests.get(f"{BASE_URL}/bootstrap-static/").json()
    
    # Map: player_id -> player object for quick lookup
    players = {p["id"]: p for p in bootstrap["elements"]}

    # Map: team_id -> team name
    teams_lookup = {t["id"]: t["name"] for t in bootstrap["teams"]}

    # Determine the current gameweek from events
    current_gw = next(event["id"] for event in bootstrap["events"] if event["is_current"])
    
    #helper functions
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
    st.write("Footballers dataframe")
    st.dataframe(df_footballers.iloc[400:450])
    
    #Form the teams table
    
    def get_player_points(player_id, gw=current_gw):
        url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
        data = requests.get(url).json()
        for match in data["history"]:
            if match["round"] == gw:
                return match["total_points"]
        return 0
    
    # Fetch league standings
    
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
        actual_captain_points = players[actual_captain_id]["event_points"]
        actual_captain_tuple = (actual_captain_name, actual_captain_points)

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
    
    # Add ranking history data to the dataframe
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

    # Make adjacent points difference column
    diff_col = df_teams["Total points"].diff().fillna(0).astype(int)
    
    # Insert diff column *after* "value"
    col_position = df_teams.columns.get_loc("Total points") + 1
    df_teams.insert(col_position, "Adjacent points difference", diff_col)
    
    st.write("Teams Dataframe")
    st.dataframe(df_teams.astype(str))

    # Metrics

    live = requests.get(f"https://fantasy.premierleague.com/api/event/{current_gw}/live/").json()
    points = [p["stats"]["total_points"] for p in live["elements"] if p["stats"]["minutes"] > 0]
    avg_points = sum(points) / len(points)
    
    # Helper functions
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

    # Metrics logic
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
        if min_value < 0:    
            rule_based_metrics.append(
                f"{team_name} got {tup[0]} in and removed {tup[1]} unwisely and saw a change of {min_value} points."
            )
        else:
            rule_based_metrics.append(
                f"{team_name} got {tup[0]} in and removed {tup[1]} and saw a change of only {min_value} points."
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
        if players.get(row["Footballer ID"], {}).get("minutes", 0) == 0:
            continue
        price_vs_points += f"{row['Footballer name']} valued at {row['Price (in Millions £)']} scored {row['GW points']} points. \n"
    rule_based_metrics.append(price_vs_points.strip())
    
    # 15 Most chosen players vs points
    most_chosen_players = df_footballers.sort_values("Times chosen in squad", ascending=False).head(5)
    chosen_vs_points = "Most picked footballers \n"
    for _, row in most_chosen_players.iterrows():
        if players.get(row["Footballer ID"], {}).get("minutes", 0) == 0:
            continue
        chosen_vs_points += f"{row['Footballer name']} chosen {row['Times chosen in squad']} times scored {row['GW points']} points. \n"
    rule_based_metrics.append(chosen_vs_points.strip())
    
    # 16 Most captained players vs points
    most_captained_players = df_footballers.sort_values("Times captained", ascending=False).head(5)
    captained_vs_points = "Most times picked as captains \n"
    for _, row in most_captained_players.iterrows():
        if players.get(row["Footballer ID"], {}).get("minutes", 0) == 0:
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
    
    all_playing_XI_players_df = pd.DataFrame(who_played_me.items(), columns= ["Footballer ID", "Played in teams"])
    all_playing_XI_players_df = df_footballers.merge(all_playing_XI_players_df, on="Footballer ID")
    
    st.write("All footballers playing for our managers")
    st.dataframe(all_playing_XI_players_df)
    
    non_zero_captains = df_footballers[df_footballers["Times captained"] > 0]
    rare_captains_cutoff = int(non_zero_captains["Times captained"].nsmallest(1).iloc[-1])
    top_scoring_rare_captains_df = all_playing_XI_players_df[all_playing_XI_players_df["Times captained"] == rare_captains_cutoff].sort_values("GW points", ascending=False)
    top_captain_cutoff = int(top_scoring_rare_captains_df["GW points"].nlargest(1).iloc[-1])
    top_scoring_rare_captains_df = top_scoring_rare_captains_df[top_scoring_rare_captains_df["GW points"] == top_captain_cutoff][["Footballer name", "GW points", "Times captained", "Played in teams"]]
    top_scoring_rare_captains_str = top_scoring_rare_captains_df.to_string(index=False)
    
    rule_based_metrics.append(f"Here are the top scoring player(s) only chosen captain {rare_captains_cutoff} times. The table consists which team(s) chose them: {top_scoring_rare_captains_str}")
    
    st.write("Rule based metrics list")
    st.write(rule_based_metrics)
    
    
    # Preprocess dataframe and metrics list to feed to LLM

    rule_based_metrics_text = "\n".join(f"- {metric}" for metric in rule_based_metrics)
    df_teams_text = df_teams.to_string(index=False)
    
    # Report Generation
    
        # Extra statistics to display on the roundup
    
    # 1. Top 3, Bottom 3
    top_3_df = df_teams.sort_values("Total points", ascending=False).head(3)[["Player name", "Team name", "Total points"]]
    top_3_str = top_3_df.to_string(index=False)
    bottom_3_df = df_teams.sort_values("Total points", ascending=False).tail(3)[["Player name", "Team name", "Total points"]]
    bottom_3_str = bottom_3_df.to_string(index=False)
    
    
    # 2. Best/Worst transfer making teams
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
    
    # 3. League Name, Gameweek Number
    league_name_text = f"League name: {league_name}"
    gw_text = f"Gameweek Number: {gw}"
    
    
    # 4. Biggest rank riser/fallers    
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
    
    
    # 5. Unique picks and their scores
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
    
    
    # 6. Captaincy effectiveness chart
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
    
    # 7. Top-3 teams with the highest single-player reliance
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

    # 8. Chips usage effectiveness

    # Aggregate chip effectiveness per entry (sum of chip impacts)
    chips_scores = {entry_id: 0 for entry_id in entries}

    for idx, entry_id in enumerate(entries):
        for gw in range(1, current_gw + 1):
            #picks_url = f"{BASE_URL}/entry/{entry_id}/event/{gw}/picks/"
            #picks = requests.get(picks_url).json()
            picks = actual_all_picks[gw][idx]

            if "picks" not in picks:  # skip if data not available
                continue

            chip_used = picks.get("active_chip") # Only 1 chip can be used per gameweek in FPL
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

                #prev_url = f"{BASE_URL}/entry/{entry_id}/event/{gw-1}/picks/"
                #prev_picks = requests.get(prev_url).json()
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

    # Convert results to DataFrame with team names
    results = []
    for entry_id, score in chips_scores.items():
        results.append({"Entry ID": entry_id, "Season chips score": score})

    df_chips = pd.DataFrame(results)
    df_chips = df_chips.merge(df_teams, on="Entry ID")
    df_chips = df_chips[["Team name", "Season chips score"]]
    
    top_3_chips_cutoff = df_chips["Season chips score"].nlargest(3).iloc[-1]
    bottom_3_chips_cutoff = df_chips["Season chips score"].nsmallest(3).iloc[-1]
    
    top_3_chips_df = df_chips[df_chips["Season chips score"] >= top_3_chips_cutoff].sort_values(by="Season chips score", ascending=False)
    top_3_chips_str = top_3_chips_df.to_string(index=False)
    bottom_3_chips_df = df_chips[df_chips["Season chips score"] <= bottom_3_chips_cutoff].sort_values(by="Season chips score", ascending=True)
    bottom_3_chips_str = bottom_3_chips_df.to_string(index=False)
    
    
    # 9. Most/Least points from captain    
    all_teams_captain_points_df = pd.DataFrame(all_teams_captain_points.items(), columns = ["Entry ID", "Total captain points overall"])
    top_3_highest_scoring_captains_cutoff = all_teams_captain_points_df["Total captain points overall"].nlargest(3).iloc[-1]
    top_3_highest_scoring_captains_df = all_teams_captain_points_df[all_teams_captain_points_df["Total captain points overall"] >= top_3_highest_scoring_captains_cutoff].sort_values("Total captain points overall", ascending=False)
    top_3_highest_scoring_captains_str = top_3_highest_scoring_captains_df.to_string(index=False)
    
    bottom_3_lowest_scoring_captains_cutoff = all_teams_captain_points_df["Total captain points overall"].nsmallest(3).iloc[-1]
    bottom_3_lowest_scoring_captains_df = all_teams_captain_points_df[all_teams_captain_points_df["Total captain points overall"] <= bottom_3_lowest_scoring_captains_cutoff].sort_values("Total captain points overall", ascending=False)
    bottom_3_lowest_scoring_captains_str = bottom_3_lowest_scoring_captains_df.to_string(index=False)
    
    
    # 10. Most/Least points by auto-substitutions
    automatic_substitutions_df = df_teams[["Team name", "Points earned by auto-substitutions"]]
    top_3_highest_scoring_autosubs_cutoff = automatic_substitutions_df["Points earned by auto-substitutions"].nlargest(3).iloc[-1]
    top_3_highest_scoring_autosubs_df = automatic_substitutions_df[automatic_substitutions_df["Points earned by auto-substitutions"] >= top_3_highest_scoring_autosubs_cutoff].sort_values("Points earned by auto-substitutions", ascending=False)
    top_3_highest_scoring_autosubs_str = top_3_highest_scoring_autosubs_df.to_string(index=False)
    
    bottom_3_lowest_scoring_autosubs_cutoff = automatic_substitutions_df["Points earned by auto-substitutions"].nsmallest(3).iloc[-1]
    bottom_3_lowest_scoring_autosubs_df = automatic_substitutions_df[automatic_substitutions_df["Points earned by auto-substitutions"] <= bottom_3_lowest_scoring_autosubs_cutoff].sort_values("Points earned by auto-substitutions", ascending=False)
    bottom_3_lowest_scoring_autosubs_str = bottom_3_lowest_scoring_autosubs_df.to_string(index=False)
    
    
    # 11. Most/Least points by Vice Captain acting as Captain
    vc_turned_c_df = df_teams[["Team name", "Additional points earned by VC acting as captain"]]
    top_3_highest_scoring_vc2c_cutoff = vc_turned_c_df["Additional points earned by VC acting as captain"].nlargest(3).iloc[-1]
    top_3_highest_scoring_vc2c_df = vc_turned_c_df[vc_turned_c_df["Additional points earned by VC acting as captain"] >= top_3_highest_scoring_vc2c_cutoff].sort_values("Additional points earned by VC acting as captain", ascending=False)
    top_3_highest_scoring_vc2c_str = top_3_highest_scoring_vc2c_df.to_string(index=False)
    
    bottom_3_lowest_scoring_vc2c_cutoff = vc_turned_c_df["Additional points earned by VC acting as captain"].nsmallest(3).iloc[-1]
    bottom_3_lowest_scoring_vc2c_df = vc_turned_c_df[vc_turned_c_df["Additional points earned by VC acting as captain"] <= bottom_3_lowest_scoring_vc2c_cutoff].sort_values("Additional points earned by VC acting as captain", ascending=False)
    bottom_3_lowest_scoring_vc2c_str = bottom_3_lowest_scoring_vc2c_df.to_string(index=False)
    
    
    # New LLM output format
    # More helper statistics
    
    df_teams_times_captained_strip = df_teams[["Player name", "Team name", "Chosen captain with times captained this GW", "Acting captain with points this GW"]]
    df_teams_times_captained_strip_str = df_teams_times_captained_strip.to_string(index=False)
    
    df_teams_times_chosen_strip = df_teams[["Player name", "Team name", "Playing XI with times chosen this GW"]]
    df_teams_times_chosen_strip_str = df_teams_times_chosen_strip.to_string(index=False)
    
    chip_usage_stats = df_teams[["Player name", "Team name", "Chips used"]]
    chip_usage_stats_str = chip_usage_stats.to_string(index=False)
    
    favourite_teams_df = df_teams[["Player name", "Team name", "Favourite team"]]
    favourite_teams_str = favourite_teams_df.to_string(index=False)
    
    standings_df = df_teams[["Player name", "Team name", "Total points"]]
    standings_str = standings_df.to_string(index=False)
    
    
    # Real Gameweek results
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
    
    
    # Stripped teams table
    df_teams_stripped = df_teams.copy()
    df_teams_stripped = df_teams_stripped.drop(columns=["Entry ID", "Playing XI with ID", "Adjacent points difference", "Transfer hits this GW", "Transfer hits overall", "Bench with ID"])
    df_teams_stripped_str = df_teams_stripped.to_string(index=False)
    
    # Rivals
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

    #Prompt
    prompt = f"""You are the witty but fair commissioner of an FPL league. Voice: Insightful, borderline mean-spirited, witty, banterful. Not very kids friendly (PG 13).
    Write about Gameweek {current_gw} in 500-700 words. You also need to have knowledge of the Premier League footballing world and make
    remarks that the fans can relate to, as a part of your sense of humor.
    
    Write your commentary in this order:
    1. Talk about most common captains chosen by the managers using this data: {df_teams_times_captained_strip_str}. Also talk about most commonly played players by every team using this data: {df_teams_times_chosen_strip_str}. Also talk about the chip usage statistics by each team: {chip_usage_stats_str}.
    *Lookout for vice captains playing as captains in their absence. In case that happens, and if the VC scores well, consider the manager lucky. Compare the chosen captain with the 'Acting captain with points this GW'column to know that.
    2. Give shoutout to managers who played rarely chosen players and scored well. Use this data: {top_unique_picks}. Also, here are rarely chosen captains and their scores: {top_scoring_rare_captains_str}. Check {chip_usage_stats} to see if anyone stood out with chip usage this week.
    3. Use the real gameweek results to trash talk about managers' favourite teams. Here are the results: {real_gameweek_results_str}. Here are managers and their favourite teams: {favourite_teams_str}.
    4. Look at these two teams. 1: {rival_1_str}, 2: {rival_2_str}. These are close to each other in terms of points and standings. Create a rivalry out of them.
    5. Looking at the rival teams, suggest what if scenarios. Eg, if team 2 didn't make their transfer, they would have had more points and be on top. Or if they played a player on their bench. Or if they captained the right player. You can also make these for the team 1, how they could have extended their lead in this rivalry.
    6. Make a closing statement about the league and the gameweek. Here is a complete table, look for interesting stats to mention in your commentary: {df_teams_stripped_str}.
    
    Other than these, also make more comment on the following stats:
    1. Unique player picks, their scores and the player name: {top_unique_picks}
    2. Players who relied the most on a single player for their scores: {top_3_reliance_str}
    3. Players who relied the least on a single player for their scores: {bottom_3_reliance_str}
    4. Top chip users throughout the season: {top_3_chips_str}
    5. Worst chip users throughout the season: {bottom_3_chips_str}
    6. Comment on how well someone chose their captain (one should choose their highest scoring player as captain always). A ratio of 1 means the best player chosen as captain. 0 means worst. Check these tables: {top_captaincy_str}, {worst_captaincy_str}.
    7. Biggest rank risers: {rise_strs}. Biggest rank fallers: {fall_strs}.
    8. Luck factor: Teams with most points from automatic-substitutions: {top_3_highest_scoring_autosubs_str}. Teams with least points from automatic-substitutions: {bottom_3_lowest_scoring_autosubs_str}.
    Teams with most points from Vice Captain acting as Captain: {top_3_highest_scoring_vc2c_str}. Teams with least points from Vice Captain acting as Captain: {bottom_3_lowest_scoring_vc2c_str}. 
    9. Use rank history to comment on the leader/loser’s season summary and performance. See if the leader is being lucky or smart.
    10. More stats can be found in this set of statements: {rule_based_metrics_text}
    In the tables above you would see player names as well, you can use them interchangeably with the team names for a more personal effect, but only sometimes.
    
    The 'Transfers' column holds transfer pair in this format: (Player IN, Player OUT, points GAINED by the transfer). So it the 3rd item is a positive number, the transfer was successful.
    *Transfer hits are the 4 point deductions a team has to face for every additional transfer than those they are allowed, use them to comment as well.
    *You also have the rankings history of every team in a list. Eg. [2, 4, 3] means a team ranked 2nd, 4th, and 3rd, after the first, second and third GW respectively. It suggests a team's rise/downfall/consistency.
    MUST talk about most common strategy throughout all the managers in our league (i.e - fielded 3 or more players almost everyone else had, captained someone who almost everyone had, used a chip that almost everyone used, etc.)
    MUST give a shoutout to one or two managers who did something unique this week (i.e - fielded a player who no one else had who scored big, captained someone who no one else did got lots of points from them, used a chip when no one else did, etc.)
    """
    
    #LLM Call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a witty fantasy football commissioner."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    # Assemble the final human-readable report
    final_response = f"""
    {league_name_text} \n
    {gw_text} \n
    Summary: \n\n
    {response.choices[0].message.content} \n\n
    Top 3 teams: \n{top_3_str} \n\n
    Bottom 3 teams: \n{bottom_3_str} \n\n
    Best Transfer Maker(s): \n{best_transfer_str} points earned in total through transfers\n\n
    Worst Transfer Maker(s): \n{worst_transfer_str} points earned in total through transfers\n\n
    Biggest rank riser(s): \n{rise_strs} \n\n
    Biggest rank faller(s): \n{fall_strs} \n\n
    Top scoring unique pick(s) chosen only {min_selected} times by managers in the league: \n {top_unique_picks} \n\n
    Best captaincy effectiveness teams: \n{top_captaincy_str} \n\n
    Worst captaincy effectiveness teams: \n{worst_captaincy_str} \n\n
    Most single-player reliant teams: \n{top_3_reliance_str} \n\n
    Least single-player reliant teams: \n{bottom_3_reliance_str} \n\n
    Teams with highest scoring captains: \n{top_3_highest_scoring_captains_str} \n\n
    Teams with lowest scoring captains: \n{bottom_3_lowest_scoring_captains_str} \n\n
    Top chips score teams: \n{top_3_chips_str} \n\n
    Bottom chips score teams: \n{bottom_3_chips_str} \n\n
    \t\tLuck Factor: \n
    Teams with most points from automatic substitutions: \n{top_3_highest_scoring_autosubs_str} \n\n
    Teams with least points from automatic substitutions: \n{bottom_3_lowest_scoring_autosubs_str} \n\n
    Teams with most points from Vice Captain acting as Captain: \n{top_3_highest_scoring_vc2c_str} \n\n
    Teams with least points from Vice Captain acting as Captain: \n{bottom_3_lowest_scoring_vc2c_str} \n\n
    """
    st.write("Report with LLM response")
    st.write(final_response)