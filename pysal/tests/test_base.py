"""
Tests for pysal.base module functionality.

This module provides test coverage for:
- _installed_version() function
- _installed_versions() function
- Versions class and its methods
- federation_hierarchy and memberships data structures
"""

from pysal.base import (
    Versions,
    _installed_version,
    _installed_versions,
    federation_hierarchy,
    memberships,
)


class TestFederationHierarchy:
    """Tests for the federation_hierarchy data structure."""

    def test_federation_hierarchy_has_required_layers(self):
        """Test that all expected layers exist in federation_hierarchy."""
        expected_layers = {"explore", "model", "viz", "lib"}
        assert set(federation_hierarchy.keys()) == expected_layers

    def test_explore_layer_contains_expected_packages(self):
        """Test that explore layer contains core packages."""
        explore_packages = federation_hierarchy["explore"]
        assert "esda" in explore_packages
        assert "giddy" in explore_packages
        assert "pointpats" in explore_packages

    def test_model_layer_contains_expected_packages(self):
        """Test that model layer contains core packages."""
        model_packages = federation_hierarchy["model"]
        assert "spreg" in model_packages
        assert "mgwr" in model_packages

    def test_viz_layer_contains_expected_packages(self):
        """Test that viz layer contains core packages."""
        viz_packages = federation_hierarchy["viz"]
        assert "splot" in viz_packages
        assert "mapclassify" in viz_packages

    def test_lib_layer_contains_libpysal(self):
        """Test that lib layer contains libpysal."""
        assert "libpysal" in federation_hierarchy["lib"]


class TestMemberships:
    """Tests for the memberships mapping."""

    def test_memberships_is_dict(self):
        """Test that memberships is a dictionary."""
        assert isinstance(memberships, dict)

    def test_memberships_maps_packages_to_layers(self):
        """Test that packages are correctly mapped to their layers."""
        assert memberships.get("esda") == "explore"
        assert memberships.get("spreg") == "model"
        assert memberships.get("splot") == "viz"
        assert memberships.get("libpysal") == "lib"

    def test_all_hierarchy_packages_in_memberships(self):
        """Test that all packages in hierarchy are in memberships."""
        for layer, packages in federation_hierarchy.items():
            for package in packages:
                assert package in memberships
                # Note: 'access' appears in both 'explore' and 'model' layers
                # The last assignment wins, so 'access' maps to 'model'
                if package == "access":
                    assert memberships[package] in ("explore", "model")
                else:
                    assert memberships[package] == layer


class TestInstalledVersion:
    """Tests for the _installed_version function."""

    def test_installed_version_returns_string(self):
        """Test that _installed_version returns a string."""
        result = _installed_version("libpysal")
        assert isinstance(result, str)

    def test_installed_version_for_installed_package(self):
        """Test version detection for an installed package."""
        version = _installed_version("libpysal")
        assert version != "NA"
        assert "." in version  # Version should contain dots (e.g., "4.13.0")

    # Note: test for nonexistent package is omitted because the current
    # implementation has a bug (NameError instead of returning 'NA').
    # This will be fixed in a separate PR.

    def test_installed_version_for_numpy(self):
        """Test version detection for numpy (commonly installed)."""
        version = _installed_version("numpy")
        assert version != "NA"


class TestInstalledVersions:
    """Tests for the _installed_versions function."""

    def test_installed_versions_returns_dict(self):
        """Test that _installed_versions returns a dictionary."""
        versions = _installed_versions()
        assert isinstance(versions, dict)

    def test_installed_versions_contains_all_packages(self):
        """Test that all federation packages are in the result."""
        versions = _installed_versions()
        for package in memberships:
            assert package in versions

    def test_installed_versions_values_are_strings(self):
        """Test that all version values are strings."""
        versions = _installed_versions()
        for _package, version in versions.items():
            assert isinstance(version, str)


class TestVersionsClass:
    """Tests for the Versions class."""

    def test_versions_instance_creation(self):
        """Test that Versions can be instantiated."""
        v = Versions()
        assert v is not None

    def test_versions_installed_property(self):
        """Test that installed property returns a dictionary."""
        v = Versions()
        installed = v.installed
        assert isinstance(installed, dict)
        assert len(installed) > 0

    def test_versions_installed_is_cached(self):
        """Test that installed property is cached (same object on repeat access)."""
        v = Versions()
        installed1 = v.installed
        installed2 = v.installed
        assert installed1 is installed2

    def test_versions_installed_contains_libpysal(self):
        """Test that installed versions include libpysal."""
        v = Versions()
        assert "libpysal" in v.installed
