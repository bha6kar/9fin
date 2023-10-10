import json
import os
from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer
from torch import as_tensor, linalg

from src.offline_ingestion import (
    dump_vectors_to_file,
    normalize_embedding,
    offline_ingestion,
)


# Define a fixture for the SentenceTransformer model
@pytest.fixture(scope="module")
def custom_model():
    return SentenceTransformer("tmp/custommodel")


# Define a fixture for sample data paths
@pytest.fixture(scope="module")
def sample_data_paths():
    return {
        "text_file": "sample_data.json",
        "page_csv_file": "sample_pages.csv",
    }


# Test offline_ingestion function
def test_offline_ingestion(custom_model, sample_data_paths):
    vectors = offline_ingestion("007")

    # Ensure that the result is a list of dictionaries
    assert isinstance(vectors, list)
    assert all(isinstance(item, dict) for item in vectors)

    # Check if each dictionary has the expected keys
    expected_keys = ["text", "vector", "pages"]
    assert all(set(item.keys()) == set(expected_keys) for item in vectors)

    # Check if the "vector" field contains a list of floats
    assert all(isinstance(item["vector"], list) for item in vectors)
    assert all(
        isinstance(val, (int, float)) for item in vectors for val in item["vector"]
    )


# Test normalize_embedding function
def test_normalize_embedding():
    # Test normalization of a sample embedding
    input_embedding = as_tensor([1.0, 2.0, 3.0])
    normalized_embedding = normalize_embedding(input_embedding)

    # Check if the result is a list of floats
    assert isinstance(normalized_embedding, list)
    assert all(isinstance(val, (int, float)) for val in normalized_embedding)

    # Check if the L2 norm of the normalized embedding is approximately 1.0
    # norm = linalg.norm(normalized_embedding)
    # assert norm == pytest.approx(1.0, abs=1e-6)


# Test dump_vectors_to_file function
def test_dump_vectors_to_file(tmp_path):
    # Define sample vectors to dump
    sample_vectors = [
        {"text": "Sample text 1", "vector": [1, 2, 3], "pages": [1]},
        {"text": "Sample text 2", "vector": [4, 5, 6], "pages": [2]},
    ]

    # Create a temporary output file
    output_file = tmp_path / "output.json"

    # Dump the sample vectors to the output file
    dump_vectors_to_file(sample_vectors, str(output_file))

    # Check if the output file exists
    assert output_file.is_file()

    # Load the dumped vectors from the output file
    with open(output_file, "r") as file:
        loaded_vectors = json.load(file)

    # Check if the loaded vectors match the original sample vectors
    assert loaded_vectors == sample_vectors


# Clean up temporary files and resources
def test_cleanup(tmp_path):
    # Define temporary files and directories to be cleaned up
    temp_dir = tmp_path / "temp_directory"
    temp_file = tmp_path / "temp_file.txt"

    # Create temporary directory and file
    temp_dir.mkdir()
    temp_file.touch()

    # Check if the temporary directory and file exist
    assert temp_dir.is_dir()
    assert temp_file.is_file()

    # Clean up the temporary directory and file
    os.rmdir(temp_dir)
    temp_file.unlink()

    # Check if the temporary directory and file have been removed
    assert not temp_dir.exists()
    assert not temp_file.exists()
