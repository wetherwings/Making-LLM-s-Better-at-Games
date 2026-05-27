#!/usr/bin/env python3
"""
Simplified Pipeline Runner
==========================
A lightweight version to test the pipeline quickly and fix issues.
"""

import os
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Test imports first
try:
    from env.tictactoe_env import TicTacToeEnv, MinimaxOpponent
    from env.connect4_env import Connect4Env, NegamaxOpponent
    from env.chopsticks_env import ChopsticksEnv
    print("Environment imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all environment files are in the env/ directory")
    exit(1)

# Create output directories
Path("graphs").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("synthetic_data").mkdir(exist_ok=True)

class SimpleGameTester:
    """Simple game testing without Tinker for initial validation"""
    
    def __init__(self):
        self.results = {}
        
    def test_game_environment(self, game_name: str, num_games: int = 10):
        """Test a game environment works correctly"""
        print(f"Testing {game_name} environment...")
        
        if game_name == "tictactoe":
            env = TicTacToeEnv(opponent=MinimaxOpponent())
        elif game_name == "connect4":
            env = Connect4Env(opponent=NegamaxOpponent(depth=3))
        elif game_name == "chopsticks":
            env = ChopsticksEnv()
        else:
            raise ValueError(f"Unknown game: {game_name}")
            
        wins = 0
        losses = 0
        draws = 0
        total_moves = 0
        
        for game in tqdm(range(num_games), desc=f"Testing {game_name}"):
            try:
                obs = env.reset()
                done = False
                moves = 0
                
                while not done and moves < 50:  # Prevent infinite loops
                    # Get a random legal move
                    legal_moves = env.legal_moves()
                    if legal_moves:
                        action = legal_moves[np.random.randint(0, len(legal_moves))]
                        result = env.step(action)
                        
                        if isinstance(result, tuple):
                            obs, reward, done = result
                        else:
                            obs = result.observation
                            reward = result.reward
                            done = result.done
                            
                        moves += 1
                    else:
                        break
                        
                # Count results
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1
                    
                total_moves += moves
                
            except Exception as e:
                print(f"Error in game {game}: {e}")
                draws += 1  # Count as draw
                
        win_rate = wins / num_games
        avg_moves = total_moves / num_games
        
        self.results[game_name] = {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate,
            'avg_moves': avg_moves
        }
        
        print(f"{game_name} Results: {wins}W-{losses}L-{draws}D, Win Rate: {win_rate:.2%}, Avg Moves: {avg_moves:.1f}")
        
    def generate_simple_graphs(self):
        """Generate simple performance graphs"""
        if not self.results:
            print("No results to graph")
            return
            
        games = list(self.results.keys())
        win_rates = [self.results[game]['win_rate'] for game in games]
        avg_moves = [self.results[game]['avg_moves'] for game in games]
        
        # Win rate graph
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.bar(games, win_rates, color=['blue', 'green', 'red'])
        ax1.set_title('Win Rates by Game')
        ax1.set_ylabel('Win Rate')
        ax1.set_ylim(0, 1)
        
        # Average moves graph
        ax2.bar(games, avg_moves, color=['blue', 'green', 'red'])
        ax2.set_title('Average Game Length')
        ax2.set_ylabel('Average Moves')
        
        plt.tight_layout()
        plt.savefig('graphs/simple_test_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Graph saved to graphs/simple_test_results.png")
        
    def save_results(self):
        """Save test results"""
        if self.results:
            with open('results/simple_test_results.json', 'w') as f:
                json.dump(self.results, f, indent=2)
            print("Results saved to results/simple_test_results.json")

def check_datasets():
    """Check if training datasets exist"""
    games = ["tictactoe", "connect4", "chopsticks"]
    
    print("Checking datasets...")
    for game in games:
        train_path = f"Dataset/{game}_train.jsonl"
        eval_path = f"Dataset/{game}_eval.jsonl"
        
        train_exists = os.path.exists(train_path)
        eval_exists = os.path.exists(eval_path)
        
        print(f"{game}: Train {'OK' if train_exists else 'MISSING'}, Eval {'OK' if eval_exists else 'MISSING'}")
        
        if train_exists:
            # Count lines
            with open(train_path, 'r') as f:
                lines = sum(1 for _ in f)
            print(f"  Training examples: {lines}")

def test_optimal_move_calculators():
    """Test the optimal move calculators from game_utils"""
    try:
        from game_utils import TicTacToeOptimal, Connect4Optimal, ChopsticksOptimal
        
        print("Testing optimal move calculators...")
        
        # Test Tic-Tac-Toe
        ttt_calc = TicTacToeOptimal()
        ttt_state = "X..|...|...O."  # Simple state
        ttt_move = ttt_calc.get_optimal_move(ttt_state)
        print(f"Tic-Tac-Toe optimal move: {ttt_move}")
        
        # Test Connect4 (simplified)
        c4_calc = Connect4Optimal(depth=2)
        c4_state = ".......\n.......\n.......\n.......\n.......\n......."
        try:
            c4_move = c4_calc.get_optimal_move(c4_state)
            print(f"Connect4 optimal move: {c4_move}")
        except Exception as e:
            print(f"Connect4 calculator error (expected): {e}")
            
        # Test Chopsticks
        chop_calc = ChopsticksOptimal(depth=3)
        chop_state = "1 1 1 1"  # Starting state
        try:
            chop_move = chop_calc.get_optimal_move(chop_state)
            print(f"Chopsticks optimal move: {chop_move}")
        except Exception as e:
            print(f"Chopsticks calculator error (expected): {e}")
            
    except ImportError as e:
        print(f"Cannot test optimal calculators: {e}")

def main():
    """Main test runner"""
    print("=" * 50)
    print("GAME AI TRAINING PIPELINE TEST")
    print("=" * 50)
    
    # Check datasets
    check_datasets()
    print()
    
    # Test optimal move calculators
    test_optimal_move_calculators()
    print()
    
    # Test game environments
    tester = SimpleGameTester()
    
    games = ["tictactoe", "connect4", "chopsticks"]
    for game in games:
        tester.test_game_environment(game, num_games=10)  # Small number for quick test
        print()
        
    # Generate graphs and save results
    tester.generate_simple_graphs()
    tester.save_results()
    
    print("=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)
    print("If all tests passed, you can run the full pipeline with:")
    print("python main_training_pipeline.py")
    print("\nMake sure to set TINKER_API_KEY environment variable first!")

if __name__ == "__main__":
    main()
