import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-pooch",
        action="store_true",
        default=False,
        help="run tests that fetch files via pooch/network",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "pooch: tests that fetch sample files via pooch/network (opt-in with --run-pooch)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-pooch"):
        return
    skip_pooch = pytest.mark.skip(reason="need --run-pooch option to run")
    for item in items:
        if "pooch" in item.keywords:
            item.add_marker(skip_pooch)
