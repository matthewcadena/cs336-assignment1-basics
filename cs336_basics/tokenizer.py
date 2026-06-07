import enum
from heapq import merge
from itertools import groupby
import regex as re

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


def _create_byte_frequencies(
    corpus: str,
    regex_pattern: str,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    """Given a raw corpus and a list of special tokens, splits the corpus by the special tokens
    and a regex pattern, then constructs a mapping from tuples of bytes in the corpus to the 
    frequency at which they appear.
    """
    corpus_split_on_special_characters = re.split(
        ('|'.join([re.escape(st) for st in special_tokens])),
        corpus,
    )
    
    pre_tokenized_segments: list[re.Scanner] = [
        re.finditer(regex_pattern, segment) 
        for segment in corpus_split_on_special_characters
    ]

    word_frequencies_per_segment: list[dict, [str, int]] = []
    for segment in pre_tokenized_segments:
        segment_word_frequencies = {}
        for word in segment:
            word_text = word.group()
            segment_word_frequencies[word_text] = segment_word_frequencies.get(word_text, 0) + 1
        word_frequencies_per_segment.append(segment_word_frequencies)

    sorted_word_frequencies_per_segment = [
        sorted(wft.items()) for wft in 
        word_frequencies_per_segment
    ]

    merged_word_frequency_groups = list[tuple[str, int]](merge(
        *sorted_word_frequencies_per_segment,
        key=lambda pair: pair[0]
    ))

    word_frequencies: dict[str, int] = {}
    for word, group in groupby(merged_word_frequency_groups, key=lambda pair: pair[0]):
        word_frequencies[word] = sum([freq for _, freq in group])
    
    byte_frequencies: dict[tuple[bytes], int] = {
        tuple([bytes([b]) for b in word.encode('utf-8')]): word_frequencies[word]
        for word in word_frequencies
    }

    return byte_frequencies


def train_bpe(
    input_path,
    vocab_size,
    special_tokens: list[str]=['<|endoftext|>']
) -> tuple[dict[int, bytes], list[tuple[tuple[bytes]]]]:
    vocab, initial_vocab_size = _create_initial_vocab(special_tokens)
    with open(input_path, "r", encoding="utf-8") as file:
        corpus = file.read()

        byte_frequencies: dict[tuple[bytes], int] = _create_byte_frequencies(
            corpus=corpus,
            regex_pattern=PAT,
            special_tokens=special_tokens,
        )

        merges = []
        num_merges = len(merges)

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
        return (vocab, merges)