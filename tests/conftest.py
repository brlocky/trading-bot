"""
Pytest configuration file
Sets up paths and common fixtures for all tests
"""

import sys
import os
import pytest
from pathlib import Path

# Add src directory to Python path for all tests
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / 'src'

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Ensure we're in the right directory
os.chdir(PROJECT_ROOT)


@pytest.fixture(scope="session")
def project_root():
    """Provide the project root directory"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def src_path():
    """Provide the src directory path"""
    return SRC_PATH


@pytest.fixture
def sample_broker():
    """Provide a clean TradingBroker instance for testing"""
    from src.environments.broker import TradingBroker
    return TradingBroker(initial_balance=100000.0)


# Configure pytest to show full diff on assertion failures
def pytest_configure(config):
    """Configure pytest settings"""
    # Set up custom markers
    config.addinivalue_line(
        "markers", "broker: mark test as a broker functionality test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add broker marker to all tests in test_broker*.py files
        if "test_broker" in item.nodeid:
            item.add_marker(pytest.mark.broker)

        # Add slow marker to performance tests
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.performance)
