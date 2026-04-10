import os
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import requests as _requests  # noqa: F401
except ModuleNotFoundError:
    fake_requests = types.ModuleType("requests")

    class FakeHTTPError(Exception):
        def __init__(self, *args, response=None):
            super().__init__(*args)
            self.response = response

    class FakeConnectionError(Exception):
        pass

    class FakeTimeout(Exception):
        pass

    class FakeSession:
        def request(self, *args, **kwargs):
            raise NotImplementedError

        def close(self):
            return None

    fake_requests.HTTPError = FakeHTTPError
    fake_requests.ConnectionError = FakeConnectionError
    fake_requests.Timeout = FakeTimeout
    fake_requests.Session = FakeSession
    sys.modules["requests"] = fake_requests

from api import APIAgent, APIClient, TransientAPIError, X_PLAYER, load_credentials, load_env_file


class EnvConfigTests(unittest.TestCase):
    def setUp(self):
        self.original_user_id = os.environ.pop("TTT_USER_ID", None)
        self.original_api_key = os.environ.pop("TTT_API_KEY", None)

    def tearDown(self):
        os.environ.pop("TTT_USER_ID", None)
        os.environ.pop("TTT_API_KEY", None)
        if self.original_user_id is not None:
            os.environ["TTT_USER_ID"] = self.original_user_id
        if self.original_api_key is not None:
            os.environ["TTT_API_KEY"] = self.original_api_key

    def test_load_env_file_sets_missing_values(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as env_file:
            env_file.write("TTT_USER_ID=1234\n")
            env_file.write('TTT_API_KEY="abc123"\n')
            env_path = env_file.name

        try:
            load_env_file(env_path)
        finally:
            os.unlink(env_path)

        self.assertEqual(os.environ["TTT_USER_ID"], "1234")
        self.assertEqual(os.environ["TTT_API_KEY"], "abc123")

    def test_load_credentials_requires_all_values(self):
        with self.assertRaisesRegex(RuntimeError, "TTT_USER_ID, TTT_API_KEY"):
            load_credentials(env_path="/tmp/nonexistent-ttt-env-file")


class APIReconnectTests(unittest.TestCase):
    def test_client_retries_http_500_and_reconnects(self):
        client = APIClient("1234", "secret")
        original_session = client._session

        error_response = mock.Mock(status_code=500)
        transient_error = sys.modules["requests"].HTTPError("server error")
        transient_error.response = error_response

        failed_response = mock.Mock()
        failed_response.raise_for_status.side_effect = transient_error

        success_response = mock.Mock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {"code": "OK", "teams": "7,8"}

        with mock.patch("api.requests.Session.request", side_effect=[failed_response, success_response]) as request_mock:
            with mock.patch("api.time.sleep") as sleep_mock:
                result = client.get_my_teams()

        self.assertEqual(result, ["7", "8"])
        self.assertEqual(request_mock.call_count, 2)
        self.assertIsNot(client._session, original_session)
        sleep_mock.assert_called_once()

    def test_agent_reloads_same_settings_after_transient_error(self):
        client = mock.Mock()
        client.get_game_details.return_value = {
            "boardsize": "9",
            "target": "5",
            "team1id": "opponent",
            "team2id": "our-team",
        }

        agent = APIAgent(
            client=client,
            game_id="game-42",
            our_team_id="our-team",
            time_limit=7.5,
            max_depth=4,
        )
        agent._load_game_details()
        original_search_agent = agent.agent
        self.assertIsNotNone(original_search_agent)
        original_search_agent._tt[("hash", "player")] = (1, 2, 3, (4, 5))

        with mock.patch("api.time.sleep"):
            agent._recover_from_server_error(TransientAPIError("HTTP 500"))
            agent._reload_game_state()

        self.assertEqual(agent.game_id, "game-42")
        self.assertEqual(agent.our_team_id, "our-team")
        self.assertEqual(agent.our_symbol, X_PLAYER)
        self.assertIsNotNone(agent.agent)
        self.assertIs(agent.agent, original_search_agent)
        self.assertEqual(agent.agent.time_limit, 7.5)
        self.assertEqual(agent.agent.max_depth, 4)
        self.assertIn(("hash", "player"), agent.agent._tt)
        client.reconnect.assert_called_once()

    def test_reconnect_state_is_isolated_per_game_runner(self):
        client1 = mock.Mock()
        client1.get_game_details.return_value = {
            "boardsize": "9",
            "target": "5",
            "team1id": "opponent-1",
            "team2id": "our-team-1",
        }
        client2 = mock.Mock()
        client2.get_game_details.return_value = {
            "boardsize": "9",
            "target": "5",
            "team1id": "opponent-2",
            "team2id": "our-team-2",
        }

        agent1 = APIAgent(client1, "game-1", "our-team-1", time_limit=5.0, max_depth=4)
        agent2 = APIAgent(client2, "game-2", "our-team-2", time_limit=5.0, max_depth=4)
        agent1._load_game_details()
        agent2._load_game_details()

        self.assertIsNot(agent1.agent, agent2.agent)
        self.assertIsNot(agent1.board, agent2.board)

        agent1.agent._tt[("only-agent-1", "X")] = (1, 0, 99, (0, 0))

        self.assertIn(("only-agent-1", "X"), agent1.agent._tt)
        self.assertNotIn(("only-agent-1", "X"), agent2.agent._tt)


if __name__ == "__main__":
    unittest.main()
