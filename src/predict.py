# my notebook is running 3.9.17 with sentence-transformers 2.2.2 torch==1.13.1
# copied these cells from my notebook - its a sentence transformer trained for sentence similarity with predict function and some data checks

from sentence_transformers import SentenceTransformer, util
from torch import as_tensor, isclose, linalg, tensor


def encode(samples):
    x = []
    model = SentenceTransformer("tmp/custommodel")
    for s in samples:
        # RVD:// not sure how to handle if the input is greater than the max input sequence length
        if len(s) > model.max_seq_length:
            s = s[: model.max_seq_length]
        vec = model.encode([s])[0]
        x.append(vec)
    return x


# a "data test"

embs = encode(["a", "a", "b"])

# RVD :// if we normalise these embeddings maybe this might speed it up on large scale using dot product?
normalized_embs = []
for emb in embs:
    norm = linalg.norm(as_tensor(emb))
    normalized_emb = emb / norm
    normalized_embs.append(normalized_emb)


same_cos = util.cos_sim(normalized_embs[0], normalized_embs[1])
diff_cos = util.cos_sim(normalized_embs[0], normalized_embs[2])
# HACK:// why does this have a rounding error - shouldn't this be determinstic?
# set_cudnn_deterministic(True)

# Specify a tolerance for the isclose function
tolerance = 1e-6

assert isclose(same_cos, tensor(1.0000), atol=tolerance)

assert diff_cos < same_cos
print(same_cos, diff_cos)
