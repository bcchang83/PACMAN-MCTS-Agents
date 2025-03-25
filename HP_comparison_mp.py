import os
import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count

def calculate_elo(red_elo, blue_elo, red_wins, blue_wins, ties, K=32):
    total_games = red_wins + blue_wins + ties
    actual_red = (red_wins + 0.5 * ties) / total_games
    expected_red = 1 / (1 + 10 ** ((blue_elo - red_elo) / 400))
    
    delta = K * (actual_red - expected_red)
    return red_elo + delta, blue_elo - delta

def run_experiment(params):
    iterations, max_depth, exploration, rollout = params
    num_games = 10
    
    print(f"Running experiment with params: {params}")
    command = f"python capture.py -r MCTSTeam_{rollout}_rollout -q -n {num_games} --redOpts \"iterations={iterations},depth={max_depth},exploration={exploration}\""
    os.system(command)
    
    # load game results
    with open("scores", "r") as f:
        lines = f.readlines()
        avg_score = float(lines[0].split(":")[1].strip())  
        red_win_rate = float(lines[2].split("(")[-1][:-2]) 
        blue_win_rate = float(lines[3].split("(")[-1][:-2])

    red_wins = red_win_rate * num_games
    blue_wins = blue_win_rate * num_games
    ties = num_games - red_wins - blue_wins
    
    # calculate elo
    red_elo, blue_elo = calculate_elo(1500, 1500, red_wins, blue_wins, ties)
    
    return [iterations, max_depth, exploration, rollout, avg_score, red_win_rate, blue_win_rate, red_elo, blue_elo]

def main():
    # HP range
    iterations_list = [50, 100, 200]
    max_depth_list = [5, 10, 20]
    exploration_factors = [1.0, 1.414, 2.0]
    rollout_policies = ["random", "heuristic"]

    # create HP space
    param_grid = list(itertools.product(iterations_list, max_depth_list, exploration_factors, rollout_policies))
    
    print(f"Total experiments: {len(param_grid)}")
    
    # Use multiprocessing to run experiments in parallel
    # Use all available cores minus 1 to keep system responsive
    num_processes = max(1, cpu_count() - 1)
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_experiment, param_grid)
    
    # Save results
    df = pd.DataFrame(results, columns=["Iterations", "MaxDepth", "ExplorationFactor", "Rollout", 
                                       "AvgScore", "RedWinRate", "BlueWinRate", "RedElo", "BlueElo"])
    df.to_csv("mcts_results.csv", index=False)
    print("Experiment finished, saved results to mcts_results.csv")

if __name__ == "__main__":
    main()