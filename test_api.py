import os
import tempfile
import unittest

from api import load_credentials, load_env_file


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


if __name__ == "__main__":
    unittest.main()
