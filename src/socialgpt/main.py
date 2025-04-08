import torch


def main():
    def get_pairs(seq: torch.Tensor) -> torch.Tensor:
        if seq.numel() < 2:
            raise ValueError("Sequence must contain at least 2 elements")
        
        return torch.stack((seq[:-1], seq[1:]), dim=1)  # shape: [len-1, 2]


    def unique_pairs_with_counts(seq: torch.Tensor) -> tuple:
        pairs = get_pairs(seq)

        # Convert to unique indices (for counting)
        base = seq.max().item() + 1  # Ensure uniqueness
        flat_ids = pairs[:, 0] * base + pairs[:, 1]

        # Count occurrences
        tensor_unique_pairs_flattened, tensor_counts = torch.unique(flat_ids, return_counts=True)

        # Recover original pairs
        tensor_unique_pairs_with_counts = torch.stack((torch.div(
            tensor_unique_pairs_flattened, base, rounding_mode='floor'
        ), tensor_unique_pairs_flattened % base, tensor_counts), dim=1)
        return tensor_unique_pairs_with_counts
    

    def replace_byte_bigrams(tensor, bigram, replacement) -> torch.Tensor:
        assert len(bigram) == 2, "Bigram must be of length 2"
        assert isinstance(replacement, int), "Replacement must be an integer"

        result = []
        i = 0
        while i < len(tensor):
            if i < len(tensor) - 1 and tensor[i] == bigram[0] and tensor[i+1] == bigram[1]:
                result.append(replacement)
                i += 2
            else:
                result.append(tensor[i].item())
                i += 1
        return torch.tensor(result, dtype=tensor.dtype)

    
    with open("/mnt/ai/tinyshakespeare/input.txt") as f:
        text = f.read()

    def encode_utf8(text: str) -> torch.Tensor:
        return torch.tensor(list(map(int, text.encode("utf-8"))))

    tensor_bytes = encode_utf8(text)
    size_before = len(tensor_bytes)

    max_vocabulary_size, max_byte_code = 260, 256
    compression_rate = 1

    tensor_pairs_with_counts = unique_pairs_with_counts(tensor_bytes)
    idx_most_common_pair = tensor_pairs_with_counts[:, 2].argmax()
    x, y, count = tensor_pairs_with_counts[idx_most_common_pair, :]

    dict_of_byte_pair_encodings = {}

    while (
        count > 1
        and max_byte_code < max_vocabulary_size
        and compression_rate < 2
    ):
        print(f"current vocabulary size: {(max_byte_code := max_byte_code + 1)}")

        print(f"bigram: {x}, {y}")
        dict_of_byte_pair_encodings[(x.item(), y.item())] = max_byte_code
        tensor_bytes = replace_byte_bigrams(tensor_bytes, (x, y), max_byte_code)
        
        compression_rate = size_before / len(tensor_bytes)
        print(f"compression rate: {compression_rate:.3f}")

        tensor_pairs_with_counts = unique_pairs_with_counts(tensor_bytes)
        idx_most_common_pair = tensor_pairs_with_counts[:, 2].argmax()
        x, y, count = tensor_pairs_with_counts[idx_most_common_pair, :]
    
    print(len(tensor_bytes))
    print(dict_of_byte_pair_encodings)

    def encode(text: str) -> torch.Tensor:
        tensor_bytes = encode_utf8(text)

        tensor_bigrams = get_pairs(tensor_bytes)

        while True:
            x, y = min(
                tensor_bigrams,
                key=lambda x: dict_of_byte_pair_encodings.get((x[0].item(), x[1].item()), float('inf'))
            )
            x, y = x.item(), y.item()

            if (x, y) not in dict_of_byte_pair_encodings:
                break
            else:
                print(f"found replacement for {(x, y)}: {dict_of_byte_pair_encodings[(x, y)]}")
                replacement = dict_of_byte_pair_encodings[(x, y)]
                tensor_bytes = replace_byte_bigrams(tensor_bytes, (x, y), replacement)
                tensor_bigrams = get_pairs(tensor_bytes)

        return tensor_bytes

    prompt = """
    The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to strings. In this lecture we build from scratch the Tokenizer used in the GPT series from OpenAI. In the process, we will see that a lot of weird behaviors and problems of LLMs actually trace back to tokenization. We'll go through a number of these issues, discuss why tokenization is at fault, and why someone out there ideally finds a way to delete this stage entirely.
    """
    print(len(prompt))
    result = encode(prompt)
    print(len(result))


if __name__ == "__main__":
    # Call the main function from the socialgpt module
    main()
