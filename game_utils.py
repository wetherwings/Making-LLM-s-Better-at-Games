"""
Game Utilities and Helper Functions
====================================
Common utilities for game environments, evaluation, and analysis.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

@dataclass
class GameResult:
    """Standardized result format for game outcomes"""
    winner: str  # "P1", "P2", or "draw"
    moves: int
    final_state: str
    regret_scores: List[float]
    time_taken: float

class OptimalMoveCalculator:
    """Base class for calculating optimal moves in games"""
    
    def __init__(self, game_type: str):
        self.game_type = game_type
        
    def get_optimal_move(self, state: str) -> str:
        """Get the optimal move for a given game state"""
        raise NotImplementedError
        
    def calculate_regret(self, actual_move: str, optimal_move: str) -> float:
        """Calculate regret for a move vs optimal"""
        if actual_move.strip() == optimal_move.strip():
            return 0.0
        return 1.0  # Binary regret for simplicity

class TicTacToeOptimal(OptimalMoveCalculator):
    """Optimal move calculator for Tic-Tac-Toe"""
    
    def __init__(self):
        super().__init__("tictactoe")
        self.win_lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        
    def get_optimal_move(self, state: str) -> str:
        """Get optimal move using minimax"""
        board = self.parse_state(state)
        best_score = float('-inf')
        best_move = None
        
        for i in range(9):
            if board[i] == '.':
                board[i] = 'X'  # Assume X is current player
                score = self.minimax(board, False, -10, 10)
                board[i] = '.'
                
                if score > best_score:
                    best_score = score
                    best_move = i
                    
        if best_move is not None:
            return str(best_move)
        return "0"  # Default
        
    def parse_state(self, state: str) -> List[str]:
        """Parse state string to board representation"""
        # Handle different state formats
        if '|' in state:
            parts = state.split('|')
            board_str = parts[-1].strip()
        else:
            board_str = state.strip()
            
        # Convert to 9-character board
        board = []
        for char in board_str:
            if char in 'XO.':
                board.append(char)
            elif char.isdigit():
                # Handle numeric representations
                board.append('.')
                
        # Pad or truncate to 9 characters
        while len(board) < 9:
            board.append('.')
        return board[:9]
        
    def minimax(self, board: List[str], is_maximizing: bool, alpha: float, beta: float) -> int:
        """Minimax algorithm with alpha-beta pruning"""
        winner = self.check_winner(board)
        
        if winner == 'X':
            return 10
        elif winner == 'O':
            return -10
        elif '.' not in board:
            return 0
            
        if is_maximizing:
            max_eval = float('-inf')
            for i in range(9):
                if board[i] == '.':
                    board[i] = 'X'
                    eval_score = self.minimax(board, False, alpha, beta)
                    board[i] = '.'
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(9):
                if board[i] == '.':
                    board[i] = 'O'
                    eval_score = self.minimax(board, True, alpha, beta)
                    board[i] = '.'
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            return min_eval
            
    def check_winner(self, board: List[str]) -> Optional[str]:
        """Check if there's a winner"""
        for line in self.win_lines:
            a, b, c = line
            if board[a] != '.' and board[a] == board[b] == board[c]:
                return board[a]
        return None

class Connect4Optimal(OptimalMoveCalculator):
    """Optimal move calculator for Connect 4"""
    
    def __init__(self, depth: int = 4):
        super().__init__("connect4")
        self.depth = depth
        self.rows = 6
        self.cols = 7
        
    def get_optimal_move(self, state: str) -> str:
        """Get optimal move using minimax with depth limit"""
        board = self.parse_state(state)
        
        best_score = float('-inf')
        best_col = 0
        
        for col in range(self.cols):
            if self.is_valid_move(board, col):
                row = self.get_next_row(board, col)
                board[row][col] = 'X'  # Assume X is current player
                
                score = self.minimax(board, self.depth - 1, False, -1000, 1000)
                board[row][col] = '.'
                
                if score > best_score:
                    best_score = score
                    best_col = col
                    
        return str(best_col)
        
    def parse_state(self, state: str) -> List[List[str]]:
        """Parse state string to Connect 4 board"""
        board = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Extract board from state string (implementation depends on state format)
        lines = state.strip().split('\n')
        for i, line in enumerate(lines[-self.rows:]):
            if i < self.rows:
                for j, char in enumerate(line[:self.cols]):
                    if char in 'XO':
                        board[i][j] = char
                        
        return board
        
    def is_valid_move(self, board: List[List[str]], col: int) -> bool:
        """Check if a column move is valid"""
        return board[0][col] == '.'
        
    def get_next_row(self, board: List[List[str]], col: int) -> int:
        """Get the next available row in a column"""
        for row in range(self.rows - 1, -1, -1):
            if board[row][col] == '.':
                return row
        return -1
        
    def minimax(self, board: List[List[str]], depth: int, is_maximizing: bool, 
                alpha: float, beta: float) -> int:
        """Minimax with depth limit"""
        if depth == 0 or self.is_terminal(board):
            return self.evaluate_board(board)
            
        if is_maximizing:
            max_eval = float('-inf')
            for col in range(self.cols):
                if self.is_valid_move(board, col):
                    row = self.get_next_row(board, col)
                    board[row][col] = 'X'
                    eval_score = self.minimax(board, depth - 1, False, alpha, beta)
                    board[row][col] = '.'
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for col in range(self.cols):
                if self.is_valid_move(board, col):
                    row = self.get_next_row(board, col)
                    board[row][col] = 'O'
                    eval_score = self.minimax(board, depth - 1, True, alpha, beta)
                    board[row][col] = '.'
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            return min_eval
            
    def is_terminal(self, board: List[List[str]]) -> bool:
        """Check if game is over"""
        return self.check_winner(board) is not None or not self.has_valid_moves(board)
        
    def check_winner(self, board: List[List[str]]) -> Optional[str]:
        """Check for Connect 4 winner"""
        # Check horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if board[row][col] != '.':
                    if (board[row][col] == board[row][col+1] == 
                        board[row][col+2] == board[row][col+3]):
                        return board[row][col]
                        
        # Check vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if board[row][col] != '.':
                    if (board[row][col] == board[row+1][col] == 
                        board[row+2][col] == board[row+3][col]):
                        return board[row][col]
                        
        # Check diagonal (top-left to bottom-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if board[row][col] != '.':
                    if (board[row][col] == board[row+1][col+1] == 
                        board[row+2][col+2] == board[row+3][col+3]):
                        return board[row][col]
                        
        # Check diagonal (bottom-left to top-right)
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if board[row][col] != '.':
                    if (board[row][col] == board[row-1][col+1] == 
                        board[row-2][col+2] == board[row-3][col+3]):
                        return board[row][col]
                        
        return None
        
    def has_valid_moves(self, board: List[List[str]]) -> bool:
        """Check if there are any valid moves"""
        return any(board[0][col] == '.' for col in range(self.cols))
        
    def evaluate_board(self, board: List[List[str]]) -> int:
        """Simple board evaluation"""
        winner = self.check_winner(board)
        if winner == 'X':
            return 100
        elif winner == 'O':
            return -100
        else:
            # Simple heuristic: count potential winning lines
            score = 0
            for row in range(self.rows):
                for col in range(self.cols):
                    if board[row][col] == 'X':
                        score += 1
                    elif board[row][col] == 'O':
                        score -= 1
            return score

class ChopsticksOptimal(OptimalMoveCalculator):
    """Optimal move calculator for Chopsticks"""
    
    def __init__(self, depth: int = 6):
        super().__init__("chopsticks")
        self.depth = depth
        
    def get_optimal_move(self, state: str) -> str:
        """Get optimal move for chopsticks state"""
        game_state = self.parse_state(state)
        
        best_score = float('-inf')
        best_move = None
        
        legal_moves = self.get_legal_moves(game_state)
        for move in legal_moves:
            new_state = self.apply_move(game_state, move)
            score = self.minimax_chopsticks(new_state, self.depth - 1, False, -100, 100)
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move or "attack left with left"
        
    def parse_state(self, state: str) -> Tuple[int, int, int, int]:
        """Parse chopsticks state string"""
        # Extract numbers from state string
        numbers = [int(x) for x in state.split() if x.isdigit()]
        if len(numbers) >= 4:
            return tuple(numbers[:4])
        return (1, 1, 1, 1)  # Default starting state
        
    def get_legal_moves(self, state: Tuple[int, int, int, int]) -> List[str]:
        """Get all legal moves for current state"""
        p1_left, p1_right, p2_left, p2_right = state
        moves = []
        
        # Attack moves
        if p1_left > 0 and p2_left > 0:
            moves.append("attack left with left")
        if p1_left > 0 and p2_right > 0:
            moves.append("attack right with left")
        if p1_right > 0 and p2_left > 0:
            moves.append("attack left with right")
        if p1_right > 0 and p2_right > 0:
            moves.append("attack right with right")
            
        # Split moves
        total_fingers = p1_left + p1_right
        if total_fingers > 0:
            for left in range(min(5, total_fingers + 1)):
                right = total_fingers - left
                if 0 <= left <= 4 and 0 <= right <= 4:
                    if (left, right) != (p1_left, p1_right):
                        moves.append(f"split {left} {right}")
                        
        return moves
        
    def apply_move(self, state: Tuple[int, int, int, int], move: str) -> Tuple[int, int, int, int]:
        """Apply a move to get new state"""
        p1_left, p1_right, p2_left, p2_right = state
        
        if move.startswith("attack"):
            parts = move.split()
            target_hand = parts[1]  # left or right
            source_hand = parts[3]  # left or right
            
            attack_fingers = p1_left if source_hand == "left" else p1_right
            
            if target_hand == "left":
                new_p2_left = (p2_left + attack_fingers) % 5
                if new_p2_left == 0:
                    new_p2_left = 0
                return (p1_left, p1_right, new_p2_left, p2_right)
            else:
                new_p2_right = (p2_right + attack_fingers) % 5
                if new_p2_right == 0:
                    new_p2_right = 0
                return (p1_left, p1_right, p2_left, new_p2_right)
                
        elif move.startswith("split"):
            parts = move.split()
            new_left = int(parts[1])
            new_right = int(parts[2])
            return (new_left, new_right, p2_left, p2_right)
            
        return state
        
    def minimax_chopsticks(self, state: Tuple[int, int, int, int], depth: int, 
                          is_maximizing: bool, alpha: float, beta: float) -> int:
        """Minimax for chopsticks"""
        if depth == 0 or self.is_terminal_chopsticks(state):
            return self.evaluate_chopsticks(state)
            
        if is_maximizing:
            max_eval = float('-inf')
            for move in self.get_legal_moves(state):
                new_state = self.apply_move(state, move)
                eval_score = self.minimax_chopsticks(new_state, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            # For opponent, swap player positions
            p1_left, p1_right, p2_left, p2_right = state
            opponent_state = (p2_left, p2_right, p1_left, p1_right)
            
            for move in self.get_legal_moves(opponent_state):
                new_opponent_state = self.apply_move(opponent_state, move)
                # Swap back for evaluation
                new_state = (new_opponent_state[2], new_opponent_state[3], 
                           new_opponent_state[0], new_opponent_state[1])
                eval_score = self.minimax_chopsticks(new_state, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
            
    def is_terminal_chopsticks(self, state: Tuple[int, int, int, int]) -> bool:
        """Check if chopsticks game is over"""
        p1_left, p1_right, p2_left, p2_right = state
        return (p1_left == 0 and p1_right == 0) or (p2_left == 0 and p2_right == 0)
        
    def evaluate_chopsticks(self, state: Tuple[int, int, int, int]) -> int:
        """Evaluate chopsticks state"""
        p1_left, p1_right, p2_left, p2_right = state
        
        if p1_left == 0 and p1_right == 0:
            return -100  # Player 1 loses
        elif p2_left == 0 and p2_right == 0:
            return 100   # Player 1 wins
        else:
            # Simple heuristic: count active hands
            p1_active = (1 if p1_left > 0 else 0) + (1 if p1_right > 0 else 0)
            p2_active = (1 if p2_left > 0 else 0) + (1 if p2_right > 0 else 0)
            return (p1_active - p2_active) * 10

class GameEvaluator:
    """Comprehensive game evaluation system"""
    
    def __init__(self):
        self.optimal_calculators = {
            "tictactoe": TicTacToeOptimal(),
            "connect4": Connect4Optimal(depth=4),
            "chopsticks": ChopsticksOptimal(depth=6)
        }
        
    def evaluate_move(self, game_type: str, state: str, move: str) -> float:
        """Evaluate a single move against optimal play"""
        if game_type not in self.optimal_calculators:
            return 0.0
            
        calculator = self.optimal_calculators[game_type]
        optimal_move = calculator.get_optimal_move(state)
        return calculator.calculate_regret(move, optimal_move)
        
    def evaluate_game_trajectory(self, game_type: str, trajectory: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate a complete game trajectory"""
        total_regret = 0
        move_count = len(trajectory)
        regret_scores = []
        
        for state, move in trajectory:
            regret = self.evaluate_move(game_type, state, move)
            total_regret += regret
            regret_scores.append(regret)
            
        return {
            "total_regret": total_regret,
            "avg_regret": total_regret / max(move_count, 1),
            "regret_scores": regret_scores,
            "perfect_moves": sum(1 for r in regret_scores if r == 0),
            "perfect_move_percentage": sum(1 for r in regret_scores if r == 0) / max(move_count, 1)
        }

class ProgressTracker:
    """Track progress with ETA and budget monitoring"""
    
    def __init__(self, total_tasks: int, budget_limit: float, time_limit: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = time.time()
        self.budget_limit = budget_limit
        self.time_limit = time_limit
        self.cost_so_far = 0.0
        
    def update(self, completed_tasks: int = 1, cost: float = 0.0):
        """Update progress tracking"""
        self.completed_tasks += completed_tasks
        self.cost_so_far += cost
        
    def get_eta(self) -> str:
        """Get estimated time remaining"""
        if self.completed_tasks == 0:
            return "Unknown"
            
        elapsed = time.time() - self.start_time
        avg_time_per_task = elapsed / self.completed_tasks
        remaining_tasks = self.total_tasks - self.completed_tasks
        eta_seconds = avg_time_per_task * remaining_tasks
        
        minutes, seconds = divmod(int(eta_seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"
        
    def get_progress_info(self) -> Dict[str, Any]:
        """Get comprehensive progress information"""
        elapsed = time.time() - self.start_time
        progress_percent = (self.completed_tasks / self.total_tasks) * 100
        
        return {
            "completed": self.completed_tasks,
            "total": self.total_tasks,
            "progress_percent": progress_percent,
            "elapsed_time": elapsed,
            "eta": self.get_eta(),
            "cost_so_far": self.cost_so_far,
            "budget_remaining": self.budget_limit - self.cost_so_far,
            "time_remaining": max(0, self.time_limit - elapsed)
        }
