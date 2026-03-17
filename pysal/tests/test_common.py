"""
Tests for pysal.lib.common module functionality.

This module provides test coverage for:
- simport() function for safe module imports
- requires() decorator for dependency checking
- jit decorator fallback when numba is not available
- Module-level constants
"""

from pysal.lib.common import (
    ATOL,
    HAS_JIT,
    MISSINGVALUE,
    RTOL,
    PatsyError,
    jit,
    requires,
    simport,
)


class TestConstants:
    """Tests for module-level constants."""

    def test_rtol_is_float(self):
        """Test that RTOL is a float."""
        assert isinstance(RTOL, float)
        assert RTOL > 0

    def test_atol_is_float(self):
        """Test that ATOL is a float."""
        assert isinstance(ATOL, float)
        assert ATOL > 0

    def test_missingvalue_is_none(self):
        """Test that MISSINGVALUE is None."""
        assert MISSINGVALUE is None

    def test_has_jit_is_boolean(self):
        """Test that HAS_JIT is a boolean."""
        assert isinstance(HAS_JIT, bool)

    def test_patsyerror_is_exception_type(self):
        """Test that PatsyError is an exception type."""
        assert issubclass(PatsyError, BaseException)


class TestSimport:
    """Tests for the simport function."""

    def test_simport_returns_tuple(self):
        """Test that simport returns a tuple."""
        result = simport("os")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_simport_success_for_stdlib_module(self):
        """Test successful import of standard library module."""
        success, module = simport("os")
        assert success is True
        assert module is not None
        assert hasattr(module, "path")

    def test_simport_success_for_numpy(self):
        """Test successful import of numpy."""
        success, module = simport("numpy")
        assert success is True
        assert module is not None
        assert hasattr(module, "array")

    def test_simport_failure_for_nonexistent_module(self):
        """Test failed import of nonexistent module."""
        success, module = simport("nonexistent_fake_module_xyz123")
        assert success is False
        assert module is None

    def test_simport_success_for_submodule(self):
        """Test successful import of submodule."""
        success, module = simport("os.path")
        assert success is True
        assert module is not None

    def test_simport_with_libpysal(self):
        """Test import of libpysal."""
        success, module = simport("libpysal")
        assert success is True
        assert module is not None


class TestJitDecorator:
    """Tests for the jit decorator (fallback when numba not available)."""

    def test_jit_decorator_returns_callable(self):
        """Test that jit decorator returns a callable."""

        @jit
        def sample_function(x):
            return x * 2

        assert callable(sample_function)

    def test_jit_decorated_function_works(self):
        """Test that jit-decorated function executes correctly."""

        @jit
        def add_numbers(a, b):
            return a + b

        result = add_numbers(2, 3)
        assert result == 5

    def test_jit_with_kwargs(self):
        """Test jit decorator with keyword arguments (numba-style)."""

        @jit(nopython=True, cache=True)
        def multiply(x, y):
            return x * y

        assert callable(multiply)
        assert multiply(3, 4) == 12

    def test_jit_preserves_function_behavior(self):
        """Test that jit doesn't alter function behavior."""

        def original(x):
            return x**2

        decorated = jit(original)
        for val in [0, 1, 2, 5, 10]:
            assert decorated(val) == original(val)


class TestRequiresDecorator:
    """Tests for the requires decorator."""

    def test_requires_with_available_module(self):
        """Test requires decorator with available module."""

        @requires("os")
        def function_needing_os():
            return "success"

        result = function_needing_os()
        assert result == "success"

    def test_requires_with_multiple_available_modules(self):
        """Test requires decorator with multiple available modules."""

        @requires("os", "sys", "json")
        def function_needing_multiple():
            return "all available"

        result = function_needing_multiple()
        assert result == "all available"

    def test_requires_with_unavailable_module(self, capsys):
        """Test requires decorator with unavailable module."""

        @requires("nonexistent_fake_module_xyz123", verbose=True)
        def function_needing_fake():
            return "should not reach here"

        # Call the passer function
        function_needing_fake()

        # Check that warning was printed
        captured = capsys.readouterr()
        assert "missing dependencies" in captured.out
        assert "nonexistent_fake_module_xyz123" in captured.out

    def test_requires_with_verbose_false(self, capsys):
        """Test requires decorator with verbose=False."""

        @requires("nonexistent_fake_module_xyz123", verbose=False)
        def silent_function():
            return "should not reach here"

        # Call the passer function
        silent_function()

        # Check that no warning was printed
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_requires_mixed_availability(self, capsys):
        """Test requires with mix of available and unavailable modules."""

        @requires("os", "nonexistent_fake_module_xyz123", verbose=True)
        def mixed_function():
            return "should not reach here"

        mixed_function()

        captured = capsys.readouterr()
        assert "missing dependencies" in captured.out
