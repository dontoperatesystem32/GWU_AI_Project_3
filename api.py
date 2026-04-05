import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from main import Board, MiniMaxAgent, O_PLAYER, X_PLAYER, EMPTY


DEFAULT_USER_ID = "3728"
DEFAULT_API_KEY = "51c5e9c02680cf99af06"

BASE_URL = "https://www.notexponential.com/aip2pgaming/api/index.php"
POLL_INTERVAL = 3
REQUEST_TIMEOUT = 15


JsonDict = Dict[str, Any]


def load_credentials() -> Tuple[str, str]:
    user_id = os.getenv("TTT_USER_ID", DEFAULT_USER_ID)
    api_key = os.getenv("TTT_API_KEY", DEFAULT_API_KEY)
    return user_id, api_key


def split_csv_ids(raw_value: str) -> List[str]:
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def parse_json_object(raw_value: Any, default: Optional[JsonDict] = None) -> JsonDict:
    if default is None:
        default = {}

    if not raw_value:
        return default
    if isinstance(raw_value, str):
        return json.loads(raw_value)
    return raw_value


class APIClient:
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

    def _get(self, params: JsonDict) -> JsonDict:
        response = requests.get(
            BASE_URL,
            headers=self._auth_headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def _post(self, data: JsonDict) -> JsonDict:
        headers = {
            **self._auth_headers,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = requests.post(
            BASE_URL,
            headers=headers,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def _check(self, result: JsonDict, context: str = "") -> JsonDict:
        if result.get("code") != "OK":
            message = result.get("message", "Unknown error")
            raise RuntimeError(f"API error [{context}]: {message}  |  Full: {result}")
        return result

    def create_team(self, name: str) -> str:
        result = self._post({"type": "team", "name": name})
        self._check(result, "create_team")
        return str(result["teamId"])

    def add_team_member(self, team_id: str, user_id: str) -> None:
        result = self._post({"type": "member", "teamId": team_id, "userId": user_id})
        self._check(result, "add_team_member")

    def remove_team_member(self, team_id: str, user_id: str) -> None:
        result = self._post(
            {"type": "removeMember", "teamId": team_id, "userId": user_id}
        )
        self._check(result, "remove_team_member")

    def get_team_members(self, team_id: str) -> List[str]:
        result = self._get({"type": "team", "teamId": team_id})
        self._check(result, "get_team_members")
        return result.get("userIds", [])

    def get_my_teams(self) -> List[str]:
        result = self._get({"type": "myTeams"})
        self._check(result, "get_my_teams")
        return split_csv_ids(result.get("teams", ""))

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

    def get_my_games(self, open_only: bool = False) -> List[str]:
        request_type = "myOpenGames" if open_only else "myGames"
        result = self._get({"type": request_type})
        self._check(result, "get_my_games")
        return split_csv_ids(result.get("games", ""))

    def get_game_details(self, game_id: str) -> JsonDict:
        result = self._get({"type": "gameDetails", "gameId": game_id})
        self._check(result, "get_game_details")
        return parse_json_object(result.get("game", "{}"))

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

    def get_moves(self, game_id: str, count: int = 50) -> List[JsonDict]:
        result = self._get({"type": "moves", "gameId": game_id, "count": count})
        self._check(result, "get_moves")
        return result.get("moves", [])

    def get_board_string(self, game_id: str) -> str:
        result = self._get({"type": "boardString", "gameId": game_id})
        self._check(result, "get_board_string")
        return result.get("output", "")

    def get_board_map(self, game_id: str) -> JsonDict:
        result = self._get({"type": "boardMap", "gameId": game_id})
        self._check(result, "get_board_map")
        return parse_json_object(result.get("output"), {})


def sync_board_from_map(board: Board, board_map: JsonDict) -> None:
    board.load_position(board_map)


def display_board_from_string(board_str: str, n: int) -> None:
    rows = board_str.strip().split("\n") if board_str.strip() else [EMPTY * n for _ in range(n)]
    col_header = "   " + "  ".join(str(col) for col in range(n))

    print(col_header)
    print("  +" + "---" * n + "+")
    for row_index, row_str in enumerate(rows):
        cells = "  ".join(row_str)
        print(f"{row_index} | {cells} |")
    print("  +" + "---" * n + "+")


class APIAgent:
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
        self._last_board_str: Optional[str] = None

    def _load_game_details(self) -> JsonDict:
        details = self.client.get_game_details(self.game_id)
        self.n = int(details.get("boardsize", 12))
        self.m = int(details.get("target", 6))
        self.our_symbol = self._determine_symbol(details)
        self.board = Board(self.n, self.m)
        self.agent = MiniMaxAgent(
            self.our_symbol,
            time_limit=self.time_limit,
            max_depth=self.max_depth,
        )

        print(f"\nGame {self.game_id} loaded:")
        print(f"Board : {self.n}×{self.n}  |  Target : {self.m}")
        print(f"We play as : {self.our_symbol}")
        return details

    def _determine_symbol(self, details: JsonDict) -> str:
        team1 = str(details.get("team1id", ""))
        team2 = str(details.get("team2id", ""))

        if self.our_team_id == team1:
            return O_PLAYER
        if self.our_team_id == team2:
            return X_PLAYER
        raise ValueError("Your team is not part of this game")

    def _fetch_game_snapshot(self) -> Tuple[JsonDict, JsonDict, str]:
        details = self.client.get_game_details(self.game_id)
        board_map = self.client.get_board_map(self.game_id)
        board_str = self.client.get_board_string(self.game_id)
        return details, board_map, board_str

    def _is_our_turn(self, details: JsonDict) -> bool:
        return str(details.get("turnteamid", "")) == self.our_team_id

    def _is_game_over(self, details: JsonDict) -> Tuple[bool, Optional[str]]:
        winner = details.get("winnerteamid")

        if winner is not None and str(winner).strip() != "":
            return True, str(winner)

        if self.board is None:
            return False, None

        terminal, board_winner = self.board.is_terminal()
        if not terminal:
            return False, None

        if board_winner is None:
            return True, None

        team1 = str(details.get("team1id", ""))
        team2 = str(details.get("team2id", ""))
        winner_team = team1 if board_winner == X_PLAYER else team2
        return True, winner_team

    def _sync_board(self, board_map: JsonDict, board_str: str) -> bool:
        if self.board is None:
            raise RuntimeError("Board not initialized")

        if board_str == self._last_board_str:
            return False

        sync_board_from_map(self.board, board_map)
        self._last_board_str = board_str
        print("\nCurrent board:")
        display_board_from_string(board_str, self.n)
        return True

    def _print_game_result(self, winner_team: Optional[str]) -> None:
        print("\n" + "═" * 50)
        if winner_team == self.our_team_id:
            print("Win!")
        elif winner_team is None:
            print("Draw!")
        else:
            print(f"Loss. Winner: team {winner_team}")
        print("═" * 50)

    def run(self) -> None:
        self._load_game_details()

        while True:
            details, board_map, board_str = self._fetch_game_snapshot()
            self._sync_board(board_map, board_str)

            over, winner_team = self._is_game_over(details)
            if over:
                self._print_game_result(winner_team)
                break

            if not self._is_our_turn(details):
                print(f"Waiting for opponent… (polling every {POLL_INTERVAL}s)")
                time.sleep(POLL_INTERVAL)
                continue

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
                move_id = self.client.make_move(self.game_id, self.our_team_id, row, col)
                print(f"Move accepted (moveId={move_id})")
            except RuntimeError as error:
                print(f"Move rejected: {error}")
                time.sleep(2)

            time.sleep(1)


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


def maybe_add_team_member(client: APIClient, team_id: str) -> None:
    answer = input("\nAdd a team member? [y/n]: ").strip().lower()
    if answer != "y":
        return

    teammate_id = input("Teammate's user ID: ").strip()
    client.add_team_member(team_id, teammate_id)
    print(f"Added user {teammate_id} to team {team_id}")


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


def read_search_settings() -> Tuple[float, int]:
    try:
        time_limit = float(
            input("Time limit per move in seconds [default 5.0]: ").strip() or "5.0"
        )
        max_depth = int(input("Max search depth [default 6]: ").strip() or "6")
        return time_limit, max_depth
    except ValueError:
        return 5.0, 6


def main() -> None:
    print("\nAI P2P Tic Tac Toe — Automated Agent Runner")

    user_id, api_key = load_credentials()
    client = APIClient(user_id, api_key)

    our_team_id = setup_team(client)
    maybe_add_team_member(client, our_team_id)
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
