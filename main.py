import random
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


Player = str
Move = Tuple[int, int]
Window = List[Player]
WindowCells = Tuple[Move, ...]
TTKey = Tuple[int, Player]
TTEntry = Tuple[int, int, int, Optional[Move]]

EMPTY = "-"
X_PLAYER = "X"
O_PLAYER = "O"
WIN_SCORE = 10**9

DIRECTIONS: Tuple[Move, ...] = ((0, 1), (1, 0), (1, 1), (1, -1))
FRONTIER_RADIUS = 2
ROOT_CANDIDATE_LIMIT = 24
SECOND_PLY_CANDIDATE_LIMIT = 18
DEEP_CANDIDATE_LIMIT = 12
DEFENSE_ORDER_BIAS = 1.25
DEADLINE_SAFETY_MARGIN = 0.05
TIME_CHECK_INTERVAL = 128
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
TT_MAX_SIZE = 200_000


def opponent(player: Player) -> Player:
    return O_PLAYER if player == X_PLAYER else X_PLAYER


def build_window_weights(m: int) -> List[int]:
    weights = [0 for _ in range(m + 1)]
    for count in range(1, m):
        score = 8**count
        if count == m - 1:
            score *= 24
        elif count == m - 2:
            score *= 6
        elif count == m - 3:
            score *= 2
        weights[count] = score
    weights[m] = WIN_SCORE // 4
    return weights


def build_center_weights(n: int) -> List[List[int]]:
    center = (n - 1) / 2
    weights: List[List[int]] = []
    for r in range(n):
        row: List[int] = []
        for c in range(n):
            distance = abs(r - center) + abs(c - center)
            row.append(max(0, int(round(n - distance))))
        weights.append(row)
    return weights


def precompute_windows(n: int, m: int) -> Tuple[List[WindowCells], List[List[List[int]]]]:
    windows: List[WindowCells] = []
    cell_to_windows: List[List[List[int]]] = [[[] for _ in range(n)] for _ in range(n)]

    def add_window(cells: Sequence[Move]) -> None:
        index = len(windows)
        window = tuple(cells)
        windows.append(window)
        for r, c in window:
            cell_to_windows[r][c].append(index)

    for r in range(n):
        for c in range(n - m + 1):
            add_window([(r, c + offset) for offset in range(m)])

    for c in range(n):
        for r in range(n - m + 1):
            add_window([(r + offset, c) for offset in range(m)])

    for r in range(n - m + 1):
        for c in range(n - m + 1):
            add_window([(r + offset, c + offset) for offset in range(m)])

    for r in range(n - m + 1):
        for c in range(m - 1, n):
            add_window([(r + offset, c - offset) for offset in range(m)])

    return windows, cell_to_windows


def line_score_from_counts(x_count: int, o_count: int, weights: Sequence[int]) -> int:
    if x_count > 0 and o_count > 0:
        return 0
    if x_count > 0:
        return weights[x_count]
    if o_count > 0:
        return -weights[o_count]
    return 0


class SearchTimeout(Exception):
    pass


class Board:
    def __init__(self, n: int, m: int):
        if n <= 0:
            raise ValueError("Board size n must be positive")
        if m <= 0 or m > n:
            raise ValueError("Win length m must satisfy 1 <= m <= n")

        self.n = n
        self.m = m
        self.grid: List[List[Player]] = [[EMPTY for _ in range(n)] for _ in range(n)]
        self.move_count = 0

        self._window_weights = build_window_weights(m)
        self._center_weights = build_center_weights(n)
        self._windows, self._cell_to_windows = precompute_windows(n, m)
        self._window_x_counts = [0 for _ in self._windows]
        self._window_o_counts = [0 for _ in self._windows]
        self._window_scores = [0 for _ in self._windows]
        self._line_balance = 0
        self._center_balance = 0

        self._frontier_support = [[0 for _ in range(n)] for _ in range(n)]
        self._frontier: Set[Move] = set()

        seed = n * 1000 + m * 17
        rng = random.Random(seed)
        self._zobrist = [
            [[rng.getrandbits(64) for _ in range(2)] for _ in range(n)]
            for _ in range(n)
        ]
        self.zobrist_hash = 0

        self._winner: Optional[Player] = None
        self._last_move: Optional[Move] = None
        self._history: List[Tuple[Optional[Player], Optional[Move]]] = []

    def reset(self) -> None:
        self.grid = [[EMPTY for _ in range(self.n)] for _ in range(self.n)]
        self.move_count = 0
        self._window_x_counts = [0 for _ in self._windows]
        self._window_o_counts = [0 for _ in self._windows]
        self._window_scores = [0 for _ in self._windows]
        self._line_balance = 0
        self._center_balance = 0
        self._frontier_support = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self._frontier.clear()
        self.zobrist_hash = 0
        self._winner = None
        self._last_move = None
        self._history.clear()

    def load_position(self, board_map: Dict[str, Player]) -> None:
        self.reset()

        occupied: List[Tuple[int, int, Player]] = []
        for key, symbol in board_map.items():
            row, col = map(int, key.split(","))
            if not (0 <= row < self.n and 0 <= col < self.n):
                raise ValueError("Board map contains out-of-bounds coordinates")
            if symbol not in (X_PLAYER, O_PLAYER):
                raise ValueError("Board map contains an invalid symbol")
            if self.grid[row][col] != EMPTY:
                raise ValueError("Board map contains duplicate coordinates")

            self.grid[row][col] = symbol
            self.move_count += 1
            self._center_balance += self._piece_sign(symbol) * self._center_weights[row][col]
            self.zobrist_hash ^= self._zobrist_value(row, col, symbol)
            occupied.append((row, col, symbol))

        for row, col, _ in occupied:
            self._update_frontier_support(row, col, 1)

        self._rebuild_window_state()
        self._winner = self._recompute_winner_from_windows()

    def make_move(self, r: int, c: int, player: Player) -> None:
        if not (0 <= r < self.n and 0 <= c < self.n):
            raise ValueError("Move out of bounds")
        if self.grid[r][c] != EMPTY:
            raise ValueError("Cell already occupied")

        self._history.append((self._winner, self._last_move))
        self.grid[r][c] = player
        self.move_count += 1
        self._center_balance += self._piece_sign(player) * self._center_weights[r][c]
        self.zobrist_hash ^= self._zobrist_value(r, c, player)

        self._frontier.discard((r, c))
        self._update_frontier_support(r, c, 1)
        self._update_windows_for_move(r, c, player, delta=1)

        self._last_move = (r, c)
        self._winner = player if self._creates_win_from_move(r, c, player) else None

    def undo_move(self, r: int, c: int) -> None:
        player = self.grid[r][c]
        if player == EMPTY:
            raise ValueError("Cell already empty")

        self.grid[r][c] = EMPTY
        self.move_count -= 1
        self._center_balance -= self._piece_sign(player) * self._center_weights[r][c]
        self.zobrist_hash ^= self._zobrist_value(r, c, player)

        self._update_frontier_support(r, c, -1)
        if self._frontier_support[r][c] > 0:
            self._frontier.add((r, c))
        else:
            self._frontier.discard((r, c))

        self._update_windows_for_move(r, c, player, delta=-1)
        self._winner, self._last_move = self._history.pop()

    def is_full(self) -> bool:
        return self.move_count == self.n * self.n

    def check_winner(self, player: Player) -> bool:
        return self._winner == player

    def is_terminal(self) -> Tuple[bool, Optional[Player]]:
        if self._winner is not None:
            return True, self._winner
        if self.is_full():
            return True, None
        return False, None

    def legal_moves(self) -> List[Move]:
        return [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if self.grid[r][c] == EMPTY
        ]

    def ordered_moves(
        self,
        player: Player,
        ply: int,
        preferred_move: Optional[Move] = None,
    ) -> List[Move]:
        if self.move_count == 0:
            moves = self._center_moves()
            if preferred_move in moves:
                return [preferred_move] + [move for move in moves if move != preferred_move]
            return moves

        candidates = list(self._frontier)
        if not candidates:
            candidates = self.legal_moves()

        move_features: List[Tuple[int, int, int, int, int, Move, bool]] = []
        forced_moves: Set[Move] = set()

        for move in candidates:
            priority, heuristic, defense_gain, attack_gain, center_gain, forced = (
                self._classify_move(move[0], move[1], player)
            )
            move_features.append(
                (priority, heuristic, defense_gain, attack_gain, center_gain, move, forced)
            )
            if forced:
                forced_moves.add(move)

        move_features.sort(reverse=True)

        ordered: List[Move] = []
        seen: Set[Move] = set()

        def add_move(move: Move) -> None:
            if move not in seen:
                ordered.append(move)
                seen.add(move)

        if preferred_move in candidates:
            add_move(preferred_move)

        for _, _, _, _, _, move, forced in move_features:
            if forced:
                add_move(move)

        target = max(self._candidate_limit(ply), len(forced_moves))
        for _, _, _, _, _, move, _ in move_features:
            if len(ordered) >= target and move not in forced_moves and move != preferred_move:
                break
            add_move(move)

        return ordered

    def evaluate(self, player: Player) -> int:
        score_from_x = self._line_balance + int(round(self._center_balance * self._center_taper()))
        return score_from_x if player == X_PLAYER else -score_from_x

    def print(self, heading: str = "\nBoard:") -> None:
        print(heading)
        for row in self.grid:
            print(" ".join(row))

    def _piece_sign(self, player: Player) -> int:
        return 1 if player == X_PLAYER else -1

    def _zobrist_value(self, r: int, c: int, player: Player) -> int:
        player_index = 0 if player == X_PLAYER else 1
        return self._zobrist[r][c][player_index]

    def _candidate_limit(self, ply: int) -> int:
        if ply == 0:
            return ROOT_CANDIDATE_LIMIT
        if ply == 1:
            return SECOND_PLY_CANDIDATE_LIMIT
        return DEEP_CANDIDATE_LIMIT

    def _center_moves(self) -> List[Move]:
        center = (self.n - 1) / 2
        best_distance: Optional[float] = None
        moves: List[Move] = []

        for r in range(self.n):
            for c in range(self.n):
                if self.grid[r][c] != EMPTY:
                    continue
                distance = abs(r - center) + abs(c - center)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    moves = [(r, c)]
                elif distance == best_distance:
                    moves.append((r, c))

        return sorted(moves)

    def _center_taper(self) -> float:
        filled_ratio = self.move_count / max(1, self.n * self.n)
        return max(0.0, 1.0 - 1.5 * filled_ratio)

    def _rebuild_window_state(self) -> None:
        self._line_balance = 0
        for index, cells in enumerate(self._windows):
            x_count = 0
            o_count = 0
            for r, c in cells:
                if self.grid[r][c] == X_PLAYER:
                    x_count += 1
                elif self.grid[r][c] == O_PLAYER:
                    o_count += 1
            self._window_x_counts[index] = x_count
            self._window_o_counts[index] = o_count
            score = line_score_from_counts(x_count, o_count, self._window_weights)
            self._window_scores[index] = score
            self._line_balance += score

    def _recompute_winner_from_windows(self) -> Optional[Player]:
        for x_count, o_count in zip(self._window_x_counts, self._window_o_counts):
            if x_count == self.m:
                return X_PLAYER
            if o_count == self.m:
                return O_PLAYER
        return None

    def _update_windows_for_move(self, r: int, c: int, player: Player, delta: int) -> None:
        for index in self._cell_to_windows[r][c]:
            old_score = self._window_scores[index]
            if player == X_PLAYER:
                self._window_x_counts[index] += delta
            else:
                self._window_o_counts[index] += delta

            new_score = line_score_from_counts(
                self._window_x_counts[index],
                self._window_o_counts[index],
                self._window_weights,
            )
            self._window_scores[index] = new_score
            self._line_balance += new_score - old_score

    def _update_frontier_support(self, r: int, c: int, delta: int) -> None:
        row_start = max(0, r - FRONTIER_RADIUS)
        row_end = min(self.n, r + FRONTIER_RADIUS + 1)
        col_start = max(0, c - FRONTIER_RADIUS)
        col_end = min(self.n, c + FRONTIER_RADIUS + 1)

        for nr in range(row_start, row_end):
            for nc in range(col_start, col_end):
                if nr == r and nc == c:
                    continue
                self._frontier_support[nr][nc] += delta
                if self.grid[nr][nc] == EMPTY:
                    if self._frontier_support[nr][nc] > 0:
                        self._frontier.add((nr, nc))
                    else:
                        self._frontier.discard((nr, nc))

    def _creates_win_from_move(self, r: int, c: int, player: Player) -> bool:
        for dr, dc in DIRECTIONS:
            total = 1
            total += self._count_direction(r, c, dr, dc, player)
            total += self._count_direction(r, c, -dr, -dc, player)
            if total >= self.m:
                return True
        return False

    def _count_direction(self, r: int, c: int, dr: int, dc: int, player: Player) -> int:
        count = 0
        nr, nc = r + dr, c + dc
        while 0 <= nr < self.n and 0 <= nc < self.n and self.grid[nr][nc] == player:
            count += 1
            nr += dr
            nc += dc
        return count

    def _classify_move(self, r: int, c: int, player: Player) -> Tuple[int, int, int, int, int, bool]:
        if self.grid[r][c] != EMPTY:
            raise ValueError("Cannot classify a non-empty cell")

        own_counts = self._window_x_counts if player == X_PLAYER else self._window_o_counts
        opp_counts = self._window_o_counts if player == X_PLAYER else self._window_x_counts

        attack_gain = 0
        defense_gain = 0
        attack_threats = 0
        defense_threats = 0
        fork_creates = 0
        fork_blocks = 0
        immediate_win = False
        immediate_block = False

        for index in self._cell_to_windows[r][c]:
            own = own_counts[index]
            opp = opp_counts[index]

            if own > 0 and opp > 0:
                continue

            if opp == 0:
                new_count = own + 1
                attack_gain += self._window_weights[new_count] - self._window_weights[own]
                if own == self.m - 1:
                    immediate_win = True
                if own == self.m - 2:
                    fork_creates += 1
                if own >= self.m - 2:
                    attack_threats += 1
            elif own == 0:
                defense_gain += self._window_weights[opp]
                if opp == self.m - 1:
                    immediate_block = True
                if opp == self.m - 2:
                    fork_blocks += 1
                if opp >= self.m - 2:
                    defense_threats += 1

        center_gain = int(round(self._center_weights[r][c] * self._center_taper()))
        heuristic = attack_gain + int(round(DEFENSE_ORDER_BIAS * defense_gain)) + center_gain

        if fork_creates >= 2:
            heuristic += self._window_weights[self.m - 1]
        if fork_blocks >= 2:
            heuristic += int(round(DEFENSE_ORDER_BIAS * self._window_weights[self.m - 1]))

        priority = 0
        if immediate_win:
            priority = 6
        elif immediate_block:
            priority = 5
        elif fork_creates >= 2:
            priority = 4
        elif fork_blocks >= 2:
            priority = 3
        elif defense_threats > 0:
            priority = 2
        elif attack_threats > 0:
            priority = 1

        forced = immediate_win or immediate_block or fork_creates >= 2 or fork_blocks >= 2
        return priority, heuristic, defense_gain, attack_gain, center_gain, forced


def iter_windows(board: Board) -> Iterable[Window]:
    for cells in board._windows:
        yield [board.grid[r][c] for r, c in cells]


def evaluate(board: Board, player: Player) -> int:
    return board.evaluate(player)


def order_moves(board: Board, player: Player = X_PLAYER, ply: int = 0) -> List[Move]:
    return board.ordered_moves(player, ply)


class MiniMaxAgent:
    def __init__(self, player: Player, time_limit: float = 2.0, max_depth: int = 6):
        self.player = player
        self.time_limit = time_limit
        self.max_depth = max_depth
        self._deadline = 0.0
        self._nodes = 0
        self._tt: Dict[TTKey, TTEntry] = {}
        self._tt_enabled = True

    def best_move(self, board: Board) -> Optional[Move]:
        terminal, _ = board.is_terminal()
        if terminal:
            return None

        legal = board.ordered_moves(self.player, 0)
        if not legal:
            return None

        reserve = min(DEADLINE_SAFETY_MARGIN, self.time_limit * 0.1)
        self._deadline = time.perf_counter() + max(0.001, self.time_limit - reserve)
        self._nodes = 0

        if len(self._tt) > TT_MAX_SIZE:
            self._tt.clear()

        best_move = legal[0]

        for move in legal:
            priority, _, _, _, _, _ = board._classify_move(move[0], move[1], self.player)
            if priority == 6:
                return move

        opponent_wins = [
            move
            for move in legal
            if board._classify_move(move[0], move[1], self.player)[0] == 5
        ]
        if len(opponent_wins) == 1:
            return opponent_wins[0]

        preferred_move: Optional[Move] = None
        for depth in range(1, self.max_depth + 1):
            if self._deadline_reached():
                break

            try:
                depth_best, _ = self._search_root(board, depth, preferred_move)
            except SearchTimeout:
                break

            if depth_best is not None:
                best_move = depth_best
                preferred_move = depth_best

        return best_move

    def _search_root(
        self,
        board: Board,
        depth: int,
        preferred_move: Optional[Move],
    ) -> Tuple[Optional[Move], int]:
        alpha = -WIN_SCORE
        beta = WIN_SCORE
        best_move: Optional[Move] = None
        best_score = -WIN_SCORE

        tt_move = self._tt_move(board.zobrist_hash, self.player)
        ordering_hint = preferred_move if preferred_move is not None else tt_move
        moves = board.ordered_moves(self.player, 0, ordering_hint)

        for move in moves:
            self._check_timeout(force=True)
            board.make_move(move[0], move[1], self.player)
            try:
                score = -self._negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    opponent(self.player),
                    ply=1,
                )
            finally:
                board.undo_move(move[0], move[1])

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

        return best_move, best_score

    def _negamax(
        self,
        board: Board,
        depth: int,
        alpha: int,
        beta: int,
        player: Player,
        ply: int,
    ) -> int:
        self._check_timeout()

        terminal, winner = board.is_terminal()
        if terminal:
            return self._terminal_score(winner, player, ply)
        if depth == 0:
            return evaluate(board, player)

        alpha_original = alpha
        beta_original = beta
        key = (board.zobrist_hash, player)
        preferred_move: Optional[Move] = None

        entry = self._tt.get(key) if self._tt_enabled else None
        if entry is not None:
            entry_depth, entry_flag, entry_score, entry_move = entry
            preferred_move = entry_move
            if entry_depth >= depth:
                if entry_flag == TT_EXACT:
                    return entry_score
                if entry_flag == TT_LOWER:
                    alpha = max(alpha, entry_score)
                elif entry_flag == TT_UPPER:
                    beta = min(beta, entry_score)
                if alpha >= beta:
                    return entry_score

        moves = board.ordered_moves(player, ply, preferred_move)
        if not moves:
            return 0

        best_score = -WIN_SCORE
        best_move: Optional[Move] = None

        for move in moves:
            board.make_move(move[0], move[1], player)
            try:
                score = -self._negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    opponent(player),
                    ply + 1,
                )
            finally:
                board.undo_move(move[0], move[1])

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        if self._tt_enabled:
            flag = TT_EXACT
            if best_score <= alpha_original:
                flag = TT_UPPER
            elif best_score >= beta_original:
                flag = TT_LOWER
            self._tt[key] = (depth, flag, best_score, best_move)

        return best_score

    def _check_timeout(self, force: bool = False) -> None:
        self._nodes += 1
        if force or self._nodes % TIME_CHECK_INTERVAL == 0:
            if self._deadline_reached():
                raise SearchTimeout

    def _deadline_reached(self) -> bool:
        return time.perf_counter() >= self._deadline

    def _terminal_score(self, winner: Optional[Player], player: Player, ply: int) -> int:
        if winner is None:
            return 0
        if winner == player:
            return WIN_SCORE - ply
        return -WIN_SCORE + ply

    def _tt_move(self, hash_value: int, player: Player) -> Optional[Move]:
        if not self._tt_enabled:
            return None
        entry = self._tt.get((hash_value, player))
        if entry is None:
            return None
        return entry[3]


class Game:
    def __init__(self, n: int, m: int, time_limit: float = 2.0, max_depth: int = 6):
        self.board = Board(n, m)
        self.agents = {
            X_PLAYER: MiniMaxAgent(X_PLAYER, time_limit, max_depth),
            O_PLAYER: MiniMaxAgent(O_PLAYER, time_limit, max_depth),
        }

    def play(self) -> None:
        current = X_PLAYER

        while True:
            self.board.print()
            print(f"AI ({current}) thinking...")

            move = self.agents[current].best_move(self.board)
            if move is None:
                print("No moves left.")
                break

            r, c = move
            self.board.make_move(r, c, current)

            terminal, winner = self.board.is_terminal()
            if terminal:
                self.board.print("\nFinal Board:")
                if winner:
                    print(f"Winner: {winner}")
                else:
                    print("Draw")
                break

            current = opponent(current)


def read_game_settings() -> Tuple[int, int]:
    try:
        n = int(input("Board size n: "))
        m = int(input("Win length m: "))
        return n, m
    except Exception:
        return 3, 3


def read_search_settings() -> Tuple[float, int]:
    try:
        time_limit = float(input("Time limit (seconds): ") or "2.0")
        max_depth = int(input("Max depth: ") or "6")
        return time_limit, max_depth
    except Exception:
        return 2.0, 6


def main() -> None:
    print("Generalized Tic Tac Toe — AI vs AI")
    n, m = read_game_settings()
    time_limit, max_depth = read_search_settings()
    game = Game(n, m, time_limit, max_depth)
    game.play()


if __name__ == "__main__":
    main()
