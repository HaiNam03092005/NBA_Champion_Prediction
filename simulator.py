import random

def log5_matchup_probability(prob_a, prob_b): # Bill James's formula Log5
    if prob_a == 0 and prob_b == 0: return 0.5
    numerator = prob_a * (1.0 - prob_b)
    denominator = prob_a * (1.0 - prob_b) + prob_b * (1.0 - prob_a)
    if denominator == 0: return 0.5
    return numerator / denominator

# The system calculates the win rate by taking 70% of the AI's prediction (overall strength) plus 30% of the actual head-to-head record during the season (to identify matchup errors)
def simulate_monte_carlo_bo7(team_a, team_b, prob_col, h2h_lookup, num_simulations=1000):
    p_a_wins_single = log5_matchup_probability(team_a[prob_col], team_b[prob_col])
    
    # Find the history of head-to-head matches (H2H)
    matchup = h2h_lookup[(h2h_lookup['Season_Year'] == team_a['Season_Year']) & 
                         (h2h_lookup['TEAM_NAME'] == team_a['TEAM_NAME']) & 
                         (h2h_lookup['OPPONENT_NAME'] == team_b['TEAM_NAME'])]
    
    if not matchup.empty:
        h2h_rate = matchup.iloc[0]['H2H_Win_Rate']
        p_a_wins_single = 0.7 * p_a_wins_single + 0.3 * h2h_rate
        
    team_a_series_wins = 0
    for _ in range(num_simulations):
        a_wins, b_wins = 0, 0
        while a_wins < 4 and b_wins < 4:
            if random.random() < p_a_wins_single: a_wins += 1
            else: b_wins += 1
        if a_wins == 4: team_a_series_wins += 1
            
    return team_a if team_a_series_wins >= (num_simulations / 2) else team_b

def simulate_conference_bracket(conf_df, prob_col_top4, prob_col_top2, h2h_lookup):
    teams = {row['Seed']: row for _, row in conf_df.iterrows()}
    
    w1_8 = simulate_monte_carlo_bo7(teams[1], teams[8], prob_col_top4, h2h_lookup)
    w4_5 = simulate_monte_carlo_bo7(teams[4], teams[5], prob_col_top4, h2h_lookup)
    w2_7 = simulate_monte_carlo_bo7(teams[2], teams[7], prob_col_top4, h2h_lookup)
    w3_6 = simulate_monte_carlo_bo7(teams[3], teams[6], prob_col_top4, h2h_lookup)
    
    w_top4_a = simulate_monte_carlo_bo7(w1_8, w4_5, prob_col_top4, h2h_lookup)
    w_top4_b = simulate_monte_carlo_bo7(w2_7, w3_6, prob_col_top4, h2h_lookup)
    
    conf_predicted_top4 = [w_top4_a['TEAM_NAME'], w_top4_b['TEAM_NAME']]
    conf_champion = simulate_monte_carlo_bo7(w_top4_a, w_top4_b, prob_col_top2, h2h_lookup)
    
    return conf_predicted_top4, conf_champion