import json
from tests.common import gpt2_bytes_to_unicode
from tokenizer import train_bpe
from datetime import datetime, time, date

def print_time(start_time, end_time) -> None:
    total_seconds = int((end_time - start_time).total_seconds())

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    print(f"Time elapsed: {formatted_time}")
    return


def run_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    vocab_output_path: str="train_bpe_vocab.json",
    merges_output_path: str="train_bpe_merges.txt",
):
    start_time = datetime.now()
    
    with open(vocab_output_path, 'w') as v, open(merges_output_path, 'w') as m:
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        end_time = datetime.now()
        gpt2_byte_encoder = gpt2_bytes_to_unicode()
        decoded_vocab = {
            ''.join(
                gpt2_byte_encoder[byte] 
                for byte in token_bytes
            )
            :
            token_id
            for token_id, token_bytes in vocab.items()
        }
        json.dump(decoded_vocab, v)

        for pair in merges:
            bytes_1 = ''.join(gpt2_byte_encoder[byte] for byte in pair[0])
            bytes_2 = ''.join(gpt2_byte_encoder[byte] for byte in pair[1])
            m.write(f'{bytes_1} {bytes_2}\n')
    
    print_time(start_time, end_time)

if __name__ == "__main__":
    run_train_bpe(
        input_path="./data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=['<|endoftext|>'],
        vocab_output_path='train_bpe_tiny_stories_train_vocab.json',
        merges_output_path='train_bpe_tiny_stories_train_merges.txt',
    )
