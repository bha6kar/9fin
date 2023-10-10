import numpy as np

# Save data
vector = [1, 2, 3, 4]
pages = [10, 20, 30, 40]
np.save("data.npy", {"vector": np.array(vector), "pages": np.array(pages)})

# Load data
loaded_data = np.load("data.npy", allow_pickle=True).item()
vector_loaded = loaded_data["vector"]
pages_loaded = loaded_data["pages"]
print(loaded_data)

import struct

# Data to be saved
vector = [1, 2, 3, 4]
pages = [10, 20, 30, 40]

# Define a file path for saving the data
file_path = "data.bin"

# Save data to a binary file
with open(file_path, "wb") as file:
    # Pack the lists of integers into binary data
    packed_vector = struct.pack("i" * len(vector), *vector)
    packed_pages = struct.pack("i" * len(pages), *pages)

    # Write the packed data to the file
    file.write(packed_vector)
    file.write(packed_pages)

# Load data from the binary file
loaded_vector = []
loaded_pages = []

with open(file_path, "rb") as file:
    # Unpack the binary data
    loaded_vector = struct.unpack("i" * len(vector), file.read(4 * len(vector)))
    loaded_pages = struct.unpack("i" * len(pages), file.read(4 * len(pages)))

print("Loaded Vector:", loaded_vector)
print("Loaded Pages:", loaded_pages)

import torch
import struct

# Sample list of dictionaries
data_list = [
    {
        "vector": torch.tensor(
            [1.0, 2.0, 3.0, 4.0]
        ),  # Replace with your actual torch.Tensor
        "pages": [10, 20, 30, 40],  # Replace with your actual list of integers
    },
    {
        "vector": torch.tensor(
            [5.0, 6.0, 7.0, 8.0]
        ),  # Replace with your actual torch.Tensor
        "pages": [50, 60, 70, 80],  # Replace with your actual list of integers
    },
]

# Define a file path for saving the data
file_path = "data.bin"

# Pack the data and write it to a binary file
with open(file_path, "wb") as file:
    for data_dict in data_list:
        # Serialize the 'vector' tensor as bytes
        tensor_bytes = data_dict["vector"].numpy().tobytes()
        # Write the length of the tensor bytes as an int (4 bytes)
        tensor_length = len(tensor_bytes)
        file.write(struct.pack("i", tensor_length))
        # Write the tensor bytes
        file.write(tensor_bytes)

        # Serialize the 'pages' list of integers as bytes
        pages_bytes = struct.pack("i" * len(data_dict["pages"]), *data_dict["pages"])
        # Write the list of integers bytes
        file.write(pages_bytes)

# To read the data back from the .bin file:
loaded_data_list = []
with open(file_path, "rb") as file:
    for _ in data_list:
        # Read the length of the tensor bytes
        tensor_length = struct.unpack("i", file.read(4))[0]
        # Read the tensor bytes and deserialize
        tensor_bytes = file.read(tensor_length)
        loaded_vector = torch.from_numpy(np.frombuffer(tensor_bytes, dtype=np.float32))

        # Read the list of integers bytes and deserialize
        loaded_pages = struct.unpack(
            "i" * len(data_dict["pages"]), file.read(4 * len(data_dict["pages"]))
        )

        # Construct the loaded dictionary
        loaded_data_dict = {
            "vector": loaded_vector,
            "pages": loaded_pages,
        }

        loaded_data_list.append(loaded_data_dict)

# Check the loaded data
print("Loaded Data List:", loaded_data_list)
