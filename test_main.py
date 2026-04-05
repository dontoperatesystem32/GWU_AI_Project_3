import copy
import unittest

from main import Board, MiniMaxAgent, X_PLAYER, O_PLAYER, evaluate


def apply_moves(board: Board, moves):
    for row, col, player in moves:
        board.make_move(row, col, player)


def board_map(board: Board):
    position = {}
    for row in range(board.n):
        for col in range(board.n):
            if board.grid[row][col] != "-":
                position[f"{row},{col}"] = board.grid[row][col]
    return position


class BoardStateTests(unittest.TestCase):
    def test_make_undo_restores_state(self):
        board = Board(6, 4)
        apply_moves(
            board,
            [
                (2, 2, X_PLAYER),
                (2, 3, O_PLAYER),
                (3, 2, X_PLAYER),
                (3, 3, O_PLAYER),
            ],
        )

        baseline = {
            "grid": copy.deepcopy(board.grid),
            "move_count": board.move_count,
            "hash": board.zobrist_hash,
            "frontier": set(board._frontier),
            "line_balance": board._line_balance,
            "center_balance": board._center_balance,
            "winner": board._winner,
            "eval_x": evaluate(board, X_PLAYER),
            "eval_o": evaluate(board, O_PLAYER),
        }

        board.make_move(1, 2, X_PLAYER)
        board.undo_move(1, 2)

        self.assertEqual(board.grid, baseline["grid"])
        self.assertEqual(board.move_count, baseline["move_count"])
        self.assertEqual(board.zobrist_hash, baseline["hash"])
        self.assertEqual(board._frontier, baseline["frontier"])
        self.assertEqual(board._line_balance, baseline["line_balance"])
        self.assertEqual(board._center_balance, baseline["center_balance"])
        self.assertEqual(board._winner, baseline["winner"])
        self.assertEqual(evaluate(board, X_PLAYER), baseline["eval_x"])
        self.assertEqual(evaluate(board, O_PLAYER), baseline["eval_o"])

    def test_load_position_matches_incremental_state(self):
        board = Board(7, 4)
        apply_moves(
            board,
            [
                (3, 3, X_PLAYER),
                (3, 4, O_PLAYER),
                (2, 3, X_PLAYER),
                (4, 4, O_PLAYER),
                (1, 3, X_PLAYER),
            ],
        )

        rebuilt = Board(7, 4)
        rebuilt.load_position(board_map(board))

        self.assertEqual(rebuilt.grid, board.grid)
        self.assertEqual(rebuilt.move_count, board.move_count)
        self.assertEqual(rebuilt.zobrist_hash, board.zobrist_hash)
        self.assertEqual(rebuilt._winner, board._winner)
        self.assertEqual(evaluate(rebuilt, X_PLAYER), evaluate(board, X_PLAYER))
        self.assertEqual(evaluate(rebuilt, O_PLAYER), evaluate(board, O_PLAYER))


class AgentBehaviorTests(unittest.TestCase):
    def test_chooses_immediate_win(self):
        board = Board(5, 4)
        apply_moves(
            board,
            [
                (2, 0, X_PLAYER),
                (0, 0, O_PLAYER),
                (2, 1, X_PLAYER),
                (0, 1, O_PLAYER),
                (2, 2, X_PLAYER),
            ],
        )

        agent = MiniMaxAgent(X_PLAYER, time_limit=0.1, max_depth=4)
        self.assertEqual(agent.best_move(board), (2, 3))

    def test_blocks_immediate_loss(self):
        board = Board(5, 4)
        apply_moves(
            board,
            [
                (0, 0, X_PLAYER),
                (2, 0, O_PLAYER),
                (1, 1, X_PLAYER),
                (2, 1, O_PLAYER),
                (0, 2, X_PLAYER),
                (2, 2, O_PLAYER),
            ],
        )

        agent = MiniMaxAgent(X_PLAYER, time_limit=0.1, max_depth=4)
        self.assertEqual(agent.best_move(board), (2, 3))

    def test_prefers_required_block_over_attack(self):
        board = Board(6, 4)
        apply_moves(
            board,
            [
                (0, 0, X_PLAYER),
                (2, 0, O_PLAYER),
                (0, 1, X_PLAYER),
                (2, 1, O_PLAYER),
                (1, 1, X_PLAYER),
                (2, 2, O_PLAYER),
            ],
        )

        agent = MiniMaxAgent(X_PLAYER, time_limit=0.1, max_depth=4)
        self.assertEqual(agent.best_move(board), (2, 3))

    def test_tt_enabled_matches_disabled(self):
        board = Board(5, 4)
        apply_moves(
            board,
            [
                (2, 2, X_PLAYER),
                (2, 1, O_PLAYER),
                (1, 2, X_PLAYER),
                (1, 1, O_PLAYER),
                (3, 2, X_PLAYER),
                (3, 1, O_PLAYER),
            ],
        )

        cached_agent = MiniMaxAgent(X_PLAYER, time_limit=0.2, max_depth=4)
        uncached_agent = MiniMaxAgent(X_PLAYER, time_limit=0.2, max_depth=4)
        uncached_agent._tt_enabled = False

        self.assertEqual(cached_agent.best_move(board), uncached_agent.best_move(board))

    def test_empty_board_returns_center_move(self):
        board = Board(12, 6)
        agent = MiniMaxAgent(X_PLAYER, time_limit=0.05, max_depth=4)
        self.assertIn(agent.best_move(board), {(5, 5), (5, 6), (6, 5), (6, 6)})

    def test_short_budget_returns_legal_move(self):
        board = Board(12, 6)
        apply_moves(
            board,
            [
                (5, 5, X_PLAYER),
                (5, 6, O_PLAYER),
                (6, 5, X_PLAYER),
                (6, 6, O_PLAYER),
                (4, 5, X_PLAYER),
                (4, 6, O_PLAYER),
            ],
        )

        agent = MiniMaxAgent(X_PLAYER, time_limit=0.01, max_depth=6)
        move = agent.best_move(board)
        self.assertIsNotNone(move)
        self.assertIn(move, board.legal_moves())


if __name__ == "__main__":
    unittest.main()
