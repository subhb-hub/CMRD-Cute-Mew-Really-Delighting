from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cmrd.io import read_json, write_json, write_npz


class AtomicIoTests(unittest.TestCase):
    def test_json_replace_retries_a_transient_windows_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_json(path, {"version": 1})
            real_replace = __import__("os").replace
            calls = 0

            def transient_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise PermissionError(5, "Access is denied")
                return real_replace(source, destination)

            with patch("cmrd.io.os.replace", side_effect=transient_replace), patch("cmrd.io.time.sleep") as sleep:
                write_json(path, {"version": 2})

            self.assertEqual(read_json(path), {"version": 2})
            self.assertEqual(calls, 2)
            sleep.assert_called_once_with(0.05)
            self.assertEqual(list(path.parent.glob(".manifest.json.*.tmp")), [])

    def test_npz_atomic_write_still_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npz"
            expected = np.arange(12, dtype=np.float32).reshape(3, 4)
            write_npz(path, features=expected)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["features"], expected)


if __name__ == "__main__":
    unittest.main()
