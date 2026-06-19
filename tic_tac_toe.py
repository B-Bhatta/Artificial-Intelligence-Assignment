"""
Interactive Tic-Tac-Toe using Minimax Algorithm

Human Player : O
AI Player    : X

Run:
    python tic_tac_toe.py

The player enters row and column values from 0 to 2.
Example: 1 2
"""

import numpy as np

HUMAN = "O"
AI = "X"
EMPTY = " " 


def create_board():
    """Create an empty 3x3 Tic-Tac-Toe board."""
    return np.full((3, 3), EMPTY, dtype=str)


def print_position_guide():
    """Display board position guide for the user."""
    print("Example board positions:")
    print("(0,0) | (0,1) | (0,2)")
    print("------+-------+------")
    print("(1,0) | (1,1) | (1,2)")
    print("------+-------+------")
    print("(2,0) | (2,1) | (2,2)")
    print()


def print_board(board):
    """Print the current board."""
    print("Current Board:")
    for i in range(3):
        print(f" {board[i][0]} | {board[i][1]} | {board[i][2]} ")
        if i < 2:
            print("---+---+---")
    print()


def check_winner(board):
    """
    Check if there is a winner.
    Returns AI, HUMAN, 'Draw', or None.
    """
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] != EMPTY:
            return row[0]

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != EMPTY:
            return board[0][col]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]

    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    # Check draw
    if not np.any(board == EMPTY):
        return "Draw"

    return None


def get_available_moves(board):
    """Return all empty cells as possible moves."""
    moves = []
    for row in range(3):
        for col in range(3):
            if board[row][col] == EMPTY:
                moves.append((row, col))
    return moves


def minimax(board, is_maximizing):
    """
    Minimax algorithm.
    AI tries to maximize score.
    Human tries to minimize score.
    """
    result = check_winner(board)

    if result == AI:
        return 1
    if result == HUMAN:
        return -1
    if result == "Draw":
        return 0

    if is_maximizing:
        best_score = -float("inf")
        for row, col in get_available_moves(board):
            board[row][col] = AI
            score = minimax(board, False)
            board[row][col] = EMPTY
            best_score = max(best_score, score)
        return best_score

    best_score = float("inf")
    for row, col in get_available_moves(board):
        board[row][col] = HUMAN
        score = minimax(board, True)
        board[row][col] = EMPTY
        best_score = min(best_score, score)
    return best_score


def find_best_move(board):
    """Find the best move for AI using Minimax."""
    best_score = -float("inf")
    best_move = None

    for row, col in get_available_moves(board):
        board[row][col] = AI
        score = minimax(board, False)
        board[row][col] = EMPTY

        if score > best_score:
            best_score = score
            best_move = (row, col)

    return best_move


def human_move(board):
    """Take and validate human input."""
    while True:
        try:
            user_input = input("Enter your move as row and column, example 1 2: ").strip()
            parts = user_input.replace(",", " ").split()

            if len(parts) != 2:
                print("Invalid input. Please enter two numbers: row column")
                continue

            row, col = map(int, parts)

            if row not in range(3) or col not in range(3):
                print("Invalid position. Row and column must be between 0 and 2.")
                continue

            if board[row][col] != EMPTY:
                print("This cell is already occupied. Choose another cell.")
                continue

            board[row][col] = HUMAN
            break

        except ValueError:
            print("Invalid input. Please enter numeric row and column values.")


def ai_move(board):
    """Perform AI move."""
    move = find_best_move(board)
    if move is not None:
        row, col = move
        board[row][col] = AI
        print(f"AI placed X at position ({row},{col})")


def play_game():
    """Main game loop."""
    board = create_board()

    print("Interactive Tic-Tac-Toe Game")
    print("Human Player: O")
    print("AI Player: X")
    print("AI uses Minimax algorithm to play optimally.")
    print()
    print_position_guide()

    # Human starts first for easier demonstration.
    while True:
        print_board(board)
        human_move(board)

        result = check_winner(board)
        if result is not None:
            print_board(board)
            display_result(result)
            break

        ai_move(board)

        result = check_winner(board)
        if result is not None:
            print_board(board)
            display_result(result)
            break


def display_result(result):
    """Display final result."""
    if result == HUMAN:
        print("Congratulations! You win.")
    elif result == AI:
        print("AI wins. Better luck next time.")
    else:
        print("The game is a draw.")


if __name__ == "__main__":
    play_game()
