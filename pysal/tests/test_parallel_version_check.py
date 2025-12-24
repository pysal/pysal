import time
import unittest
from unittest.mock import MagicMock, patch

from pysal.base import _installed_version, _installed_versions, memberships


class TestParallelVersionCheck(unittest.TestCase):

    def test_installed_version_single_package(self):
        version = _installed_version("numpy")
        self.assertIsInstance(version, str)
        self.assertNotEqual(version, "NA")

    def test_installed_version_missing_package(self):
        version = _installed_version("nonexistent_package_xyz123")
        self.assertEqual(version, "NA")

    @patch('pysal.base.importlib.import_module')
    def test_parallel_execution_faster_than_sequential(self, mock_import):
        import os
        if 'PYTEST_XDIST_WORKER' in os.environ:
            self.skipTest("Parallel execution disabled in pytest-xdist workers")

        def slow_import(_name):
            time.sleep(0.1)
            mock_mod = MagicMock()
            mock_mod.__version__ = "1.0.0"
            return mock_mod

        mock_import.side_effect = slow_import

        test_packages = list(memberships.keys())[:10]

        start = time.time()
        results = {}
        for pkg in test_packages:
            results[pkg] = _installed_version(pkg)
        sequential_time = time.time() - start

        start = time.time()
        _installed_versions()
        parallel_time = time.time() - start

        msg = (f"Parallel ({parallel_time:.2f}s) should be <50% "
               f"of sequential ({sequential_time:.2f}s)")
        self.assertLess(parallel_time, sequential_time * 0.5, msg)

    def test_installed_versions_returns_dict(self):
        versions = _installed_versions()
        self.assertIsInstance(versions, dict)
        self.assertGreater(len(versions), 0)
        for pkg, ver in versions.items():
            self.assertIsInstance(pkg, str)
            self.assertIsInstance(ver, str)

    def test_installed_versions_includes_all_packages(self):
        versions = _installed_versions()
        missing = [pkg for pkg in memberships if pkg not in versions]
        self.assertEqual(missing, [],
                        f"Missing packages in versions: {missing}")


if __name__ == "__main__":
    unittest.main()
