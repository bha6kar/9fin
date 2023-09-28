from pathlib import Path

from flask import Flask, jsonify
from sentence_transformers import SentenceTransformer

from util import get_page_result

app = Flask(__name__)


def get_item_id(doc_id: str) -> dict:
    """Gets the page result for the given document ID.

    Args:
        doc_id (str): The document ID.

    Returns:
        dict: A dictionary containing the page result, or an empty dictionary if the
            document ID is not found.
    """
    try:
        # Load the document text and page numbers.
        text_file_path = Path(f"data/texts_{doc_id}.json")
        page_csv_path = Path(f"data/pages_{doc_id}.csv")

        # Load the sentence transformer model.
        model = SentenceTransformer("tmp/custommodel")

        result = get_page_result(text_file_path, page_csv_path, model)

        return result
    except Exception as e:
        logger.error(f"Error processing document ID {doc_id}: {str(e)}")
        return {}


@app.route("/api/<doc_id>", methods=["GET"])
def get_item_id_handler(doc_id: str) -> jsonify:
    """Gets the page result for the given document ID.

    Args:
        doc_id (str): The document ID.

    Returns:
        jsonify: A JSON response containing the page result, or a 404 Not Found response
            if the document ID is not found.
    """
    try:
        result = get_item_id(doc_id)

        if not result:
            return jsonify({"message": "Item not found"}), 404
        else:
            return jsonify(result)
    except Exception as e:
        logger.error(f"Error handling request for document ID {doc_id}: {str(e)}")
        return jsonify({"message": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5007)
