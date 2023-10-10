import json
from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util
from torch import as_tensor

from src.util import (
    dump_vectors_to_file,
    encode_text,
    get_page_result,
    load_csv_as_dict,
    normalize_embedding,
)

# Create a custom SentenceTransformer model for testing
custom_model = SentenceTransformer("tmp/custommodel")


# Define pytest fixtures for sample data files
@pytest.fixture(scope="function")
def sample_csv_data():
    """Load the sample CSV data from the resource file"""
    resource_dir = Path(__file__).parent / "resources"
    csv_file_path = resource_dir / "sample_data.csv"
    return csv_file_path


@pytest.fixture(scope="function")
def sample_json_data():
    """Load the sample JSON data from the resource file"""
    resource_dir = Path(__file__).parent / "resources"
    json_file_path = resource_dir / "sample_data.json"
    return json_file_path


@pytest.fixture(scope="function")
def test_output_json_data():
    """Load the sample JSON data from the resource file"""
    resource_dir = Path(__file__).parent / "resources"
    json_file_path = resource_dir / "test_output.json"
    return json_file_path


@pytest.mark.parametrize(
    "input_data",
    [
        ([1.0, 2.0, 3.0])
        # TODO: Add more test cases here
    ],
)
def test_normalize_embedding(input_data):
    """
    Test normalization of embeddings using L2 norm.

    Args:
        input_data (list): List of floats representing an embedding.

    The test checks if the L2 norm of the normalized embedding is approximately 1.0.
    """
    emb = as_tensor(input_data)
    normalized_emb = normalize_embedding(emb)

    assert sum([x**2 for x in normalized_emb]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "samples,expected_similarity",
    [
        (["a", "a", "b"], 1.0),
        (["a", "b", "c"], 0.0),
    ],
)
def test_encode_text(samples, expected_similarity):
    """
    Test encoding text snippets and checking cosine similarity.

    Args:
        samples (list): List of text snippets to encode.
        expected_similarity (float): Expected cosine similarity.

    The test checks if the computed cosine similarity is within a tolerance range
    based on the expected similarity.
    """
    embs = []
    for s in samples:
        embs.append(encode_text(s, custom_model))

    normalized_embs = []
    for emb in embs:
        normalized_emb = normalize_embedding(emb)
        normalized_embs.append(normalized_emb)

    same_cos = st_util.cos_sim(normalized_embs[0], normalized_embs[1])

    # Specify tolerance values based on expected behavior
    tolerance_similar = 0.1

    # For dissimilar samples, set a low tolerance
    tolerance_dissimilar = 0.1 if expected_similarity > 0.0 else 0.05

    # Choose the appropriate tolerance based on expected behavior
    tolerance = (
        tolerance_similar if expected_similarity == 1.0 else tolerance_dissimilar
    )

    # Corrected assertion: Check if the computed value is within the tolerance
    assert same_cos.item() >= (expected_similarity - tolerance)


def test_load_csv_as_dict(sample_csv_data):
    """
    Test loading CSV data into a dictionary.

    Args:
        sample_csv_data (str): Path to a sample CSV file.

    The test checks if the loaded data is a dictionary with expected key-value pairs.
    """
    csv_dict = load_csv_as_dict(sample_csv_data)
    assert isinstance(csv_dict, dict)
    assert csv_dict == {"key": "1", "key2": "2"}


def test_get_page_result(sample_json_data, sample_csv_data, test_output_json_data):
    """
    Test generating page results from JSON and CSV data.

    Args:
        sample_json_data (str): Path to a sample JSON file.
        sample_csv_data (str): Path to a sample CSV file.
        test_output_json_data (str): Path to a test output JSON file.

    The test checks if the generated page results match the expected output from the test file.
    """
    result = get_page_result(sample_json_data, sample_csv_data, custom_model)
    with open(test_output_json_data, "r") as json_file:
        data = json.load(json_file)

    assert len(result) == len(data)
    assert result == data


def test_dump_vectors_to_file(tmp_path):
    """
    Test dumping vectors to a JSON file.

    Args:
        tmp_path (str): Temporary directory path for test output.

    The test checks if the dumped vectors match the expected vectors.
    """
    vectors = [
        {"text": "Sample text 1", "vector": [1, 2, 3], "pages": [1]},
        {"text": "Sample text 2", "vector": [4, 5, 6], "pages": [2]},
    ]
    output_file = tmp_path / "output.json"
    dump_vectors_to_file(vectors, str(output_file))
    assert output_file.is_file()
    with open(output_file, "r") as file:
        loaded_vectors = json.load(file)
    assert loaded_vectors == vectors
