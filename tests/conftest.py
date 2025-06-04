"""Test configuration and fixtures."""
import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

from src.config import Config
from src.config import setup_logging


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config() -> Config:
    """Create a sample configuration for testing."""
    return Config(
        vocab_size=100,
        n_embd=64,
        n_head=4,
        n_layer=2,
        block_size=32,
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=4,
        max_iters=100,
        eval_interval=10,
        eval_iters=5,
        device="cpu"
    )


@pytest.fixture
def sample_text_data() -> str:
    """Create sample text data for testing."""
    return """This is a sample text for testing.
It contains multiple sentences and paragraphs.
We use this data to test our tokenizer and model.
The text should be long enough to create meaningful tokens.
Hello world! This is another sentence.
Testing, testing, one two three."""


@pytest.fixture
def vocab_files(temp_dir: Path) -> Dict[str, Path]:
    """Create sample vocabulary files for testing."""
    encoder_file = temp_dir / "encoder.txt"
    decoder_file = temp_dir / "decoder.txt"

    # Create expanded vocabulary that covers more words in sample_text_data
    vocab = [
        "hello",
        "world",
        "test",
        "sample",
        "text",
        "the",
        "a",
        "an",
        "is",
        "this",
        "for",
        "testing",
        "it",
        "contains",
        "multiple",
        "sentences",
        "and",
        "paragraphs",
        "we",
        "use",
        "data",
        "to",
        "our",
        "tokenizer",
        "model",
        "should",
        "be",
        "long",
        "enough",
        "create",
        "meaningful",
        "tokens",
        "another",
        "sentence",
        "one",
        "two",
        "three"]

    with open(encoder_file, 'w') as f:
        for i, word in enumerate(vocab):
            f.write(f"{word} {i}\n")

    with open(decoder_file, 'w') as f:
        for i, word in enumerate(vocab):
            f.write(f"{i} {word}\n")

    return {"encoder": encoder_file, "decoder": decoder_file}


@pytest.fixture
def mock_model_state() -> Dict[str, Any]:
    """Create a mock model state dictionary."""
    return {
        "model_state_dict": {
            "token_embedding_table.weight": torch.randn(100, 64),
            "position_embedding_table.weight": torch.randn(32, 64),
            "ln_f.weight": torch.ones(64),
            "ln_f.bias": torch.zeros(64),
        },
        "config": {
            "vocab_size": 100,
            "n_embd": 64,
            "n_head": 4,
            "n_layer": 2,
            "block_size": 32,
            "dropout": 0.1,
        },
        "optimizer_state_dict": {},
        "epoch": 5,
        "train_loss": 2.5,
        "val_loss": 3.0,
    }


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    setup_logging(level="DEBUG")


@pytest.fixture
def device() -> str:
    """Get the appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def small_model_config() -> Config:
    """Create a very small model configuration for fast testing."""
    return Config(
        vocab_size=50,
        n_embd=32,
        n_head=2,
        n_layer=1,
        block_size=16,
        dropout=0.0,
        learning_rate=1e-3,
        batch_size=2,
        max_iters=10,
        eval_interval=5,
        eval_iters=2,
        device="cpu"
    )


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory."""
    return Path(__file__).parent / "data"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark GPU tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)

        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
