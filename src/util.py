"""Common util file for online and offline inference"""
import csv
import json
from pathlib import Path
from typing import List, Union

from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import Tensor, as_tensor, linalg


def normalize_embedding(emb: List[float]) -> List[float]:
    """Normalizes an embedding using the L2 norm.

    Args:
        emb (tensor): The embedding to normalize.

    Returns:
        tensor: The normalized embedding.
    """
    norm = linalg.norm(as_tensor(emb))
    normalized_emb = emb / norm
    return normalized_emb.tolist()  # Convert the tensor to a list


def encode_text(
    text: str, model: SentenceTransformer
) -> Union[List[Tensor], ndarray, Tensor]:
    """Encodes a text snippet using a sentence transformer model.

    Args:
        text (str): The text snippet to encode.

    Returns:
        tensor: The embedding of the text snippet.
    """
    if len(text) > model.max_seq_length:
        text = text[: model.max_seq_length]
    vector = model.encode([text])[0]
    return vector


def load_csv_as_dict(csv_file: str) -> dict:
    """Converts a CSV file to a dictionary of 2 columns.

    Args:
        csv_file (str): The path to the CSV file.

    Returns:
        dict: A dictionary of the CSV data with 2 columns.
    """

    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        csv_dict = {}
        for row in reader:
            csv_dict[row[0]] = row[1]
    return csv_dict


def get_page_result(
    text_file_path: Path, page_csv_path: Path, model: SentenceTransformer
) -> List[dict]:
    """Offline ingestion function.

    Args:
      doc_id (str): The document identifier.

    Returns:
      list: A list of vectors with their relevant metadata, in the following
        structure:

          {
            'text': str,
            'vector': list,
            'pages': list[int]
          }
    """

    # Load the document text and page numbers.

    with open(text_file_path, "r") as file:
        text_file = json.load(file)

    page_dict = load_csv_as_dict(page_csv_path)

    results = []

    for item in text_file:
        text = item["text"]
        vector = encode_text(text, model)
        norm_vector = normalize_embedding(vector)
        pages = [
            int(page_dict.get(key, 0)) for key in item["pages"] if key in page_dict
        ]
        results.append({"text": text, "vector": norm_vector, "pages": pages})

    return results


def dump_vectors_to_file(vectors: List[dict], output_file: str) -> None:
    """Dumps a list of vectors to a file.

    Args:
      vectors (list): A list of vectors with their relevant metadata, in the
        following structure:

          {
            'text': str,
            'vector': list,
            'pages': list[int]
          }
      output_file (str): The path to the output file.
    """
    with open(output_file, "w") as file:
        json.dump(vectors, file, indent=4)
