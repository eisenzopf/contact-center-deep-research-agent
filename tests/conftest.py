import pytest
import os
import json

@pytest.fixture
def sample_intents():
    return [
        {"name": "cancel my subscription please"},
        {"name": "how do I upgrade my plan?"},
        {"name": "need to terminate service"},
        {"name": "billing question about invoice"},
        {"name": "want to end membership"},
        {"name": "technical support needed"},
        {"name": "change payment method"},
        {"name": "how to pause subscription"}
    ]

@pytest.fixture
def cancellation_examples():
    return [
        "cancel my subscription",
        "want to terminate service",
        "how do I end my membership",
        "need to stop my account",
        "discontinue service"
    ]

@pytest.fixture
def billing_examples():
    return [
        "update billing info",
        "change payment method",
        "update credit card",
        "modify payment details"
    ]

def pytest_addoption(parser):
    parser.addoption(
        "--llm-debug",
        action="store_true",
        default=False,
        help="show LLM input and output"
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm_debug: mark test to show LLM input/output"
    )

@pytest.fixture
def llm_debug(request):
    """Debug fixture for LLM tests."""
    has_marker = request.node.get_closest_marker('llm_debug') is not None
    has_option = request.config.getoption('--llm-debug')
    debug_enabled = has_marker or has_option
    print(f"\nLLM Debug Status: marker={has_marker}, option={has_option}, enabled={debug_enabled}")
    return debug_enabled
