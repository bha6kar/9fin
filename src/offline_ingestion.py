"""Offline ingress"""
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

from util import dump_vectors_to_file, get_page_result


def offline_ingestion(doc_id: str) -> List[dict]:
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
    text_file_path = Path(f"data/texts_{doc_id}.json")
    page_csv_path = Path(f"data/pages_{doc_id}.csv")

    # Load the sentence transformer model.
    model = SentenceTransformer("tmp/custommodel")

    result = get_page_result(text_file_path, page_csv_path, model)

    return result


if __name__ == "__main__":
    document_id = "12345"
    vectors = offline_ingestion(document_id)
    output_file = "output/output_file.json"
    dump_vectors_to_file(vectors, output_file)
