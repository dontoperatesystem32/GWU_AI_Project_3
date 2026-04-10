import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from main import Board, MiniMaxAgent, O_PLAYER, X_PLAYER, EMPTY


BASE_URL = "https://www.notexponential.com/aip2pgaming/api/index.php"
ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
POLL_INTERVAL = 3
REQUEST_TIMEOUT = 15
REQUEST_RETRY_LIMIT = 20
REQUEST_RETRY_DELAY = 1.5
RECONNECT_DELAY = 2.0
SNAPSHOT_RETRY_DELAY = 0.2
TRANSIENT_HTTP_STATUS_CODES = {500}
REQUIRED_CREDENTIALS = ("TTT_USER_ID", "TTT_API_KEY")


JsonDict = Dict[str, Any]
BoardSignature = Tuple[Tuple[str, str], ...]


# Raised when the remote API fails in a way that should be retried instead of ending the game.
class TransientAPIError(RuntimeError):
    pass


# Loads key-value pairs from a local .env file without overriding exported env vars.
def load_env_file(path: str = ENV_FILE) -> None:
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key.startswith("export "):
                key = key[len("export ") :].strip()
            if not key or key in os.environ:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            os.environ[key] = value


# Loads required API credentials from the process environment or local .env file.
def load_credentials(env_path: str = ENV_FILE) -> Tuple[str, str]:
    load_env_file(env_path)

    missing = [name for name in REQUIRED_CREDENTIALS if not os.getenv(name)]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing required API credentials: {names}. "
            "Create a .env file from .env.example or export them before running api.py."
        )

    return os.environ["TTT_USER_ID"], os.environ["TTT_API_KEY"]


# Splits comma-separated API id lists while ignoring whitespace and empty entries.
def split_csv_ids(raw_value: str) -> List[str]:
    return [value.strip() for value in raw_value.split(",") if value.strip()]


# Normalizes API fields that may already be dictionaries or JSON-encoded strings.
def parse_json_object(raw_value: Any, default: Optional[JsonDict] = None) -> JsonDict:
    if default is None:
        default = {}

    if not raw_value:
        return default
    if isinstance(raw_value, str):
        return json.loads(raw_value)
    return raw_value


# Thin wrapper around the NotExponential Tic Tac Toe API endpoints used by the runner.
class APIClient:
    # Stores user credentials and common request headers for all API calls.
    def __init__(self, user_id: str, api_key: str):
        self.user_id = user_id
        self._auth_headers = {
            "x-api-key": api_key,
            "userid": user_id,
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        self._session = requests.Session()

    # Recreates the HTTP session after transient server failures or dropped connections.
    def reconnect(self) -> None:
        self._session.close()
        self._session = requests.Session()

    # Sends one authenticated HTTP request with retries for transient server failures.
    def _request(
        self,
        method: str,
        *,
        params: Optional[JsonDict] = None,
        data: Optional[JsonDict] = None,
    ) -> JsonDict:
        action = (params or data or {}).get("type", method.lower())
        headers = self._auth_headers
        if method == "POST":
            headers = {
                **headers,
                "Content-Type": "application/x-www-form-urlencoded",
            }

        last_error: Optional[Exception] = None
        for attempt in range(REQUEST_RETRY_LIMIT + 1):
            try:
                response = self._session.request(
                    method,
                    BASE_URL,
                    headers=headers,
                    params=params,
                    data=data,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as error:
                status_code = (
                    error.response.status_code if error.response is not None else None
                )
                if status_code not in TRANSIENT_HTTP_STATUS_CODES:
                    raise
                last_error = error
            except (requests.ConnectionError, requests.Timeout) as error:
                last_error = error

            if attempt == REQUEST_RETRY_LIMIT:
                break

            self.reconnect()
            time.sleep(REQUEST_RETRY_DELAY * (attempt + 1))

        raise TransientAPIError(
            f"Transient API failure [{action}] after {REQUEST_RETRY_LIMIT + 1} attempts"
        ) from last_error

    # Sends a GET request with shared auth, timeout handling, and JSON decoding.
    def _get(self, params: JsonDict) -> JsonDict:
        return self._request("GET", params=params)

    # Sends a form-encoded POST request with shared auth, timeout handling, and JSON decoding.
    def _post(self, data: JsonDict) -> JsonDict:
        return self._request("POST", data=data)

    # Converts non-OK API responses into RuntimeError messages with context.
    def _check(self, result: JsonDict, context: str = "") -> JsonDict:
        if result.get("code") != "OK":
            message = result.get("message", "Unknown error")
            raise RuntimeError(f"API error [{context}]: {message}  |  Full: {result}")
        return result

    # Creates a new team and returns its API team id.
    def create_team(self, name: str) -> str:
        result = self._post({"type": "team", "name": name})
        self._check(result, "create_team")
        return str(result["teamId"])

    # Adds a user to an existing team.
    def add_team_member(self, team_id: str, user_id: str) -> None:
        result = self._post({"type": "member", "teamId": team_id, "userId": user_id})
        self._check(result, "add_team_member")

    # Removes a user from an existing team.
    def remove_team_member(self, team_id: str, user_id: str) -> None:
        result = self._post(
            {"type": "removeMember", "teamId": team_id, "userId": user_id}
        )
        self._check(result, "remove_team_member")

    # Fetches the API user ids currently assigned to a team.
    def get_team_members(self, team_id: str) -> List[str]:
        result = self._get({"type": "team", "teamId": team_id})
        self._check(result, "get_team_members")
        return result.get("userIds", [])

    # Fetches the team ids associated with the authenticated user.
    def get_my_teams(self) -> List[str]:
        result = self._get({"type": "myTeams"})
        self._check(result, "get_my_teams")
        return split_csv_ids(result.get("teams", ""))

    # Creates a generalized Tic Tac Toe game between two teams.
    def create_game(
        self,
        team_id1: str,
        team_id2: str,
        board_size: int = 12,
        target: int = 6,
    ) -> str:
        result = self._post(
            {
                "type": "game",
                "teamId1": team_id1,
                "teamId2": team_id2,
                "gameType": "TTT",
                "boardSize": board_size,
                "target": target,
            }
        )
        self._check(result, "create_game")
        return str(result["gameId"])

    # Lists all games for the user, or only open games when requested.
    def get_my_games(self, open_only: bool = False) -> List[str]:
        request_type = "myOpenGames" if open_only else "myGames"
        result = self._get({"type": request_type})
        self._check(result, "get_my_games")
        return split_csv_ids(result.get("games", ""))

    # Retrieves one game's metadata as a dictionary.
    def get_game_details(self, game_id: str) -> JsonDict:
        result = self._get({"type": "gameDetails", "gameId": game_id})
        self._check(result, "get_game_details")
        return parse_json_object(result.get("game", "{}"))

    # Submits a move to the API and returns the accepted move id when present.
    def make_move(self, game_id: str, team_id: str, row: int, col: int) -> str:
        result = self._post(
            {
                "type": "move",
                "gameId": game_id,
                "teamId": team_id,
                "move": f"{row},{col}",
            }
        )
        self._check(result, "make_move")
        return str(result.get("moveId", ""))

    # Retrieves recent moves for one game.
    def get_moves(self, game_id: str, count: int = 50) -> List[JsonDict]:
        result = self._get({"type": "moves", "gameId": game_id, "count": count})
        self._check(result, "get_moves")
        return result.get("moves", [])

    # Retrieves the API's printable board-string representation for display.
    def get_board_string(self, game_id: str) -> str:
        result = self._get({"type": "boardString", "gameId": game_id})
        self._check(result, "get_board_string")
        return result.get("output", "")

    # Retrieves the sparse board map used to synchronize the local Board object.
    def get_board_map(self, game_id: str) -> JsonDict:
        result = self._get({"type": "boardMap", "gameId": game_id})
        self._check(result, "get_board_map")
        return parse_json_object(result.get("output"), {})


# Applies a sparse API board map to the local cached Board state.
def sync_board_from_map(board: Board, board_map: JsonDict) -> None:
    board.load_position(board_map)


# Converts a sparse board map into a stable fingerprint for change detection.
def board_signature(board_map: JsonDict) -> BoardSignature:
    return tuple(sorted((str(key), str(value)) for key, value in board_map.items()))


# Builds a printable board string directly from the sparse board map used by the AI.
def board_string_from_map(board_map: JsonDict, n: int) -> str:
    rows = [[EMPTY for _ in range(n)] for _ in range(n)]

    for key, symbol in board_map.items():
        row, col = map(int, str(key).split(","))
        rows[row][col] = symbol

    return "\n".join("".join(row) for row in rows)


# Prints the API board string with row and column labels for easier manual tracking.
def display_board_from_string(board_str: str, n: int) -> None:
    rows = board_str.strip().split("\n") if board_str.strip() else [EMPTY * n for _ in range(n)]
    col_header = "   " + "  ".join(str(col) for col in range(n))

    print(col_header)
    print("  +" + "---" * n + "+")
    for row_index, row_str in enumerate(rows):
        cells = "  ".join(row_str)
        print(f"{row_index} | {cells} |")
    print("  +" + "---" * n + "+")


# Coordinates API polling, local board synchronization, and automated move submission.
class APIAgent:
    # Stores API/game identifiers and search configuration for the remote runner.
    def __init__(
        self,
        client: APIClient,
        game_id: str,
        our_team_id: str,
        time_limit: float = 5.0,
        max_depth: int = 6,
    ):
        self.client = client
        self.game_id = game_id
        self.our_team_id = our_team_id
        self.time_limit = time_limit
        self.max_depth = max_depth

        self.our_symbol = ""
        self.n = 0
        self.m = 0
        self.board: Optional[Board] = None
        self.agent: Optional[MiniMaxAgent] = None
        self._last_board_signature: Optional[BoardSignature] = None

    # Reconnects to the same game while preserving per-agent search state when possible.
    def _recover_from_server_error(self, error: TransientAPIError) -> None:
        print(f"\nTransient server failure: {error}")
        print(
            "Reconnecting with the same game and search settings "
            f"(gameId={self.game_id}, teamId={self.our_team_id}, "
            f"time_limit={self.time_limit}, max_depth={self.max_depth})"
        )
        self.client.reconnect()
        self._last_board_signature = None
        time.sleep(RECONNECT_DELAY)

    # Keeps trying to reload the remote game until the server responds again.
    def _reload_game_state(self) -> None:
        while True:
            try:
                self._load_game_details()
                return
            except TransientAPIError as error:
                self._recover_from_server_error(error)

    # Loads game metadata and initializes the local board and minimax agent.
    def _load_game_details(self) -> JsonDict:
        details = self.client.get_game_details(self.game_id)
        next_n = int(details.get("boardsize", 12))
        next_m = int(details.get("target", 6))
        next_symbol = self._determine_symbol(details)

        reuse_agent = (
            self.agent is not None
            and self.agent.player == next_symbol
            and self.agent.time_limit == self.time_limit
            and self.agent.max_depth == self.max_depth
            and self.n == next_n
            and self.m == next_m
        )

        self.n = next_n
        self.m = next_m
        self.our_symbol = next_symbol
        self.board = Board(self.n, self.m)
        if not reuse_agent:
            self.agent = MiniMaxAgent(
                self.our_symbol,
                time_limit=self.time_limit,
                max_depth=self.max_depth,
            )

        print(f"\nGame {self.game_id} loaded:")
        print(f"Board : {self.n}×{self.n}  |  Target : {self.m}")
        print(f"We play as : {self.our_symbol}")
        return details

    # Determines our piece symbol from the game's team ordering.
    def _determine_symbol(self, details: JsonDict) -> str:
        team1 = str(details.get("team1id", ""))
        team2 = str(details.get("team2id", ""))

        if self.our_team_id == team1:
            return O_PLAYER
        if self.our_team_id == team2:
            return X_PLAYER
        raise ValueError("Your team is not part of this game")

    # Fetches a stable board/details snapshot so the displayed and searched position match.
    def _fetch_game_snapshot(self) -> Tuple[JsonDict, JsonDict]:
        while True:
            board_map_before = self.client.get_board_map(self.game_id)
            details = self.client.get_game_details(self.game_id)
            board_map_after = self.client.get_board_map(self.game_id)

            if board_signature(board_map_before) == board_signature(board_map_after):
                return details, board_map_after

            time.sleep(SNAPSHOT_RETRY_DELAY)

    # Checks whether the API says it is currently our team's turn.
    def _is_our_turn(self, details: JsonDict) -> bool:
        return str(details.get("turnteamid", "")) == self.our_team_id

    # Determines whether either the API or local board state has reached a final result.
    def _is_game_over(self, details: JsonDict) -> Tuple[bool, Optional[str]]:
        winner = details.get("winnerteamid")

        # Prefer the server's declared winner when it is available.
        if winner is not None and str(winner).strip() != "":
            return True, str(winner)

        if self.board is None:
            return False, None

        terminal, board_winner = self.board.is_terminal()
        if not terminal:
            return False, None

        if board_winner is None:
            return True, None

        # Convert the local X/O winner back into the API's team id convention.
        team1 = str(details.get("team1id", ""))
        team2 = str(details.get("team2id", ""))
        winner_team = team1 if board_winner == X_PLAYER else team2
        return True, winner_team

    # Synchronizes the local board only when the remote sparse position has changed.
    def _sync_board(self, board_map: JsonDict) -> bool:
        if self.board is None:
            raise RuntimeError("Board not initialized")

        current_signature = board_signature(board_map)
        if current_signature == self._last_board_signature:
            return False

        sync_board_from_map(self.board, board_map)
        self._last_board_signature = current_signature
        print("\nCurrent board:")
        display_board_from_string(board_string_from_map(board_map, self.n), self.n)
        return True

    # Prints a concise win/loss/draw message when the remote game ends.
    def _print_game_result(self, winner_team: Optional[str]) -> None:
        print("\n" + "═" * 50)
        if winner_team == self.our_team_id:
            print("Win!")
        elif winner_team is None:
            print("Draw!")
        else:
            print(f"Loss. Winner: team {winner_team}")
        print("═" * 50)

    # Main polling loop: wait for our turn, compute a move, submit it, and repeat.
    def run(self) -> None:
        self._reload_game_state()

        while True:
            try:
                # Each loop pulls both metadata and board state so turn/result checks match the board.
                details, board_map = self._fetch_game_snapshot()
                self._sync_board(board_map)

                over, winner_team = self._is_game_over(details)
                if over:
                    self._print_game_result(winner_team)
                    break

                if not self._is_our_turn(details):
                    print(f"Waiting for opponent… (polling every {POLL_INTERVAL}s)")
                    time.sleep(POLL_INTERVAL)
                    continue

                # From here on, the remote API says it is our turn, so the local minimax result
                # can be submitted immediately unless the game changed during the request.
                if self.agent is None or self.board is None:
                    raise RuntimeError("Agent not initialized")

                print(f"Computing move as {self.our_symbol}…")
                move = self.agent.best_move(self.board)
                if move is None:
                    print("Agent returned no move. Stopping.")
                    break

                row, col = move
                print(f"Submitting move: ({row}, {col})")

                try:
                    # Rejected moves usually mean the remote state changed; keep polling instead of exiting.
                    move_id = self.client.make_move(self.game_id, self.our_team_id, row, col)
                    print(f"Move accepted (moveId={move_id})")
                except RuntimeError as error:
                    print(f"Move rejected: {error}")
                    time.sleep(2)

                time.sleep(1)
            except TransientAPIError as error:
                self._recover_from_server_error(error)
                self._reload_game_state()


# Lets the user choose an existing team or create a new one for the API game.
def setup_team(client: APIClient) -> str:
    print("\nTeam setup:")
    existing = client.get_my_teams()
    if existing:
        print(f"Your existing teams: {', '.join(existing)}")

    choice = input("Enter existing teamId (or press Enter to create a new team): ").strip()
    if choice:
        return choice

    name = input("New team name: ").strip()
    team_id = client.create_team(name)
    print(f"Team created: {team_id}")
    client.add_team_member(team_id, client.user_id)
    print(f"Added yourself (userId={client.user_id}) to team")
    return team_id

# Lets the user create a new game or connect the runner to an existing game id.
def setup_game(client: APIClient, our_team_id: str) -> str:
    print("\nGame setup:")
    print("1. Create a new game")
    print("2. Join an existing game (you already have gameId)")
    choice = input("Choose [1/2]: ").strip()

    if choice == "2":
        return input("Enter gameId: ").strip()

    opp_team_id = input("Opponent teamId: ").strip()
    try:
        n = int(input("Board size n [default 12]: ").strip() or "12")
        m = int(input(f"Win length m [default 6, max {n}]: ").strip() or "6")
    except ValueError:
        n, m = 12, 6

    game_id = client.create_game(our_team_id, opp_team_id, n, m)
    print(f"Game created: {game_id}")
    return game_id


# Reads search controls for the remote agent, using robust defaults on bad input.
def read_search_settings() -> Tuple[float, int]:
    try:
        time_limit = float(
            input("Time limit per move in seconds [default 5.0]: ").strip() or "5.0"
        )
        max_depth = int(input("Max search depth [default 6]: ").strip() or "6")
        return time_limit, max_depth
    except ValueError:
        return 5.0, 6


# Entry point for the automated NotExponential API runner.
def main() -> None:
    print("\nAI P2P Tic Tac Toe — Automated Agent Runner")

    user_id, api_key = load_credentials()
    client = APIClient(user_id, api_key)

    our_team_id = setup_team(client)
    game_id = setup_game(client, our_team_id)

    print("\nAI settings:")
    time_limit, max_depth = read_search_settings()

    runner = APIAgent(
        client=client,
        game_id=game_id,
        our_team_id=our_team_id,
        time_limit=time_limit,
        max_depth=max_depth,
    )
    runner.run()


if __name__ == "__main__":
    main()
