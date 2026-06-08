from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO, Counter
import regex as re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _get_most_frequent_pair(
    byte_frequencies: dict[tuple[bytes], int]
) -> tuple[tuple[bytes]]:
    pairs = {}
    for bytes, freq in byte_frequencies.items():
        byte_list = list(bytes)
        for i in range(len(byte_list)-1):
            pair = (bytes[i], bytes[i+1])
            pairs[pair] = pairs.get(pair, 0) + freq
    
    sorted_pairs = sorted(
        pairs.items(),
        key=lambda p: (p[1], p[0]), # sort by count first, then break ties lexicographically
        reverse=True
    )
    if len(sorted_pairs) == 0:
        return ()
    return sorted_pairs[0][0]

def _update_byte_frequencies(
    most_frequent_pair: tuple[bytes],
    merged_pair: bytes,
    byte_frequencies: dict[tuple[bytes], int],
) -> dict[tuple[bytes], int]:
    """Given a pair of byte-sequences that we wish to merge and a frequency table mapping byte-sequences
    to the number of times they appear in our corpus, returns the updated frequency table after
    making the merge.
    """
    updated_frequencies = {}
    for bytes, freq in byte_frequencies.items():
        i, new_bytes = 0, []
        bytes_length = len(bytes)
        while i < bytes_length:
            if i+1 == bytes_length:
                new_bytes.append(bytes[i])
                break
            candidate_pair = (bytes[i], bytes[i+1])
            if candidate_pair == most_frequent_pair:
                new_bytes.append(merged_pair)
                i += 1
            else:
                new_bytes.append(candidate_pair[0])
            i += 1
        new_bytes = tuple(new_bytes)
        updated_frequencies[new_bytes] = freq
    
    return updated_frequencies   

def _create_initial_vocab(
    special_tokens: list[str]
) -> tuple[dict[int, bytes], int]:
    """Given a list of special tokens, create an initial mapping from vocab
    indices to byte sequences for each special tokens. Returns the dictionary
    itself and the number of initial entries.
    """
    num_special_tokens = len(special_tokens)
    vocab = {}
    vocab = { 
        index: bytes([index-num_special_tokens])
        for index in range(num_special_tokens, 256+num_special_tokens)
    }

    for i, special_token in enumerate[str](special_tokens):
        vocab[i] = special_token.encode('utf-8')

    return vocab, len(vocab)

def _get_chunk_word_frequencies(
    file_chunk: str,
    regex_pattern: str,
    special_tokens: list[str],
) -> Counter[str]:
    """Given a file chunk, a regex pattern, and list of special tokens for splitting,
    complete the pre-tokenization word-frequency table construction for the chunk.
    Call this on separate chunks of a file using dedicated processes to do pre-
    tokenization in parallel
    """
    file_chunk_split_on_special_characters = re.split(
        ('|'.join([re.escape(st) for st in special_tokens])),
        file_chunk,
    )
    
    pre_tokenized_segments: list[re.Scanner] = [
        re.finditer(regex_pattern, segment) 
        for segment in file_chunk_split_on_special_characters
    ]

    word_frequencies_per_segment = [
        Counter([word.group() for word in segment])
        for segment in tqdm(
            pre_tokenized_segments, desc="Intra-chunk per-segment word frequency construction")
    ]
    return word_frequencies_per_segment


def _create_byte_frequencies(
    file: BinaryIO,
    regex_pattern: str,
    special_tokens: list[str],
    num_processes: int,
) -> dict[tuple[bytes], int]:
    """Given a raw corpus and a list of special tokens, splits the corpus by the special tokens
    and a regex pattern, then constructs a mapping from tuples of bytes in the corpus to the 
    frequency at which they appear.
    """
    boundaries = find_chunk_boundaries(file, num_processes, special_tokens[0].encode('utf-8'))

    start_end_pairs = zip(boundaries[:-1], boundaries[1:])

    file_chunks = []
    for start, end in start_end_pairs:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
        file_chunks.append(chunk)

    num_chunks = len(file_chunks)
    chunk_regex_patterns = [regex_pattern] * num_chunks
    chunk_special_tokens = [special_tokens] * num_chunks

    # Pre-tokenize (obtain word counts) each chunk in parallel
    with ProcessPoolExecutor(num_processes) as executor:
        word_frequencies_per_segment_per_chunk = list(
            executor.map(
                _get_chunk_word_frequencies,
                file_chunks,
                chunk_regex_patterns,
                chunk_special_tokens,
            )
        )

    # Each chunk is still be broken up into segments, need to flatten
    word_frequencies_per_chunk = [
        word_frequencies
        for segment in word_frequencies_per_segment_per_chunk
        for word_frequencies in segment
    ]

    word_frequencies = Counter()
    for word_frequency_table in tqdm(
        word_frequencies_per_chunk,
        desc="Word frequency accumulation across segments"
    ):
        word_frequencies.update(word_frequency_table)
    
    byte_frequencies: dict[tuple[bytes], int] = {
        tuple([bytes([b]) for b in word.encode('utf-8')]): word_frequencies[word]
        for word in tqdm(word_frequencies, desc="Byte frequency construction")
    }

    return byte_frequencies


def train_bpe(
    input_path,
    vocab_size,
    special_tokens: list[str]=['<|endoftext|>']
) -> tuple[dict[int, bytes], list[tuple[tuple[bytes]]]]:
    vocab, initial_vocab_size = _create_initial_vocab(special_tokens)
    with open(input_path, "rb") as file:

        byte_frequencies: dict[tuple[bytes], int] = _create_byte_frequencies(
            file=file,
            regex_pattern=PAT,
            special_tokens=special_tokens,
            num_processes=cpu_count(),
        )

        merges = []
        num_merges = len(merges)

        progress_bar = tqdm(total=vocab_size, desc=f"Merges (out of expected {vocab_size})")

        while True:
            most_frequent_pair = _get_most_frequent_pair(byte_frequencies)
            if len(most_frequent_pair) == 0:
                break
            
            merges.append(most_frequent_pair)
            merged_pair = most_frequent_pair[0] + most_frequent_pair[1]
            vocab[initial_vocab_size + num_merges] = merged_pair

            num_merges += 1
            
            byte_frequencies = _update_byte_frequencies(
                most_frequent_pair=most_frequent_pair,
                merged_pair=merged_pair,
                byte_frequencies=byte_frequencies,
            )
            if len(vocab) == vocab_size:
                break

            progress_bar.update()
        
        progress_bar.close()

        return (vocab, merges)