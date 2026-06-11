from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from heapq import heappop, heappush
import itertools
from typing import Counter
import regex as re
from tqdm import tqdm
from multiprocessing import cpu_count

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = cpu_count()
NUM_CORPUS_CHUNKS = 98

class UpdateHeap:
    def __init__(self, input_list: list):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-pair>'
        self.counter = itertools.count()

        for pair, priority in input_list:
            self.add_pair(pair, priority)
        

    def add_pair(self, pair, priority=0):
        'Add a new pair or update the priority of an existing pair'
        if pair in self.entry_finder:
            self.remove_pair(pair)
        count = next(self.counter)
        sentinel_value = 1 # will always cause a min-heap loss since all negated bytes <= 0
        tiebreaker = (
            tuple([-b for b in pair[0]] + [sentinel_value]),
            tuple([-b for b in pair[1]] + [sentinel_value])
        ) # lexicographically break ties
        entry = [(-priority, tiebreaker), count, pair]
        self.entry_finder[pair] = entry
        heappush(self.pq, entry)

    def remove_pair(self, pair):
        'Mark an existing pair as REMOVED. Ignore if not found.'
        if pair in self.entry_finder:
            entry = self.entry_finder[pair]
            entry[-1] = self.REMOVED

    def adjust_pair_priority(self, pair, adjustment):
        '''Adjust the priority of a pair by a specified amount. 
        Use the adjustment as the base priority if not found'''
        if pair in self.entry_finder:
            [(priority, _), _, _] = self.entry_finder[pair]
            self.remove_pair(pair)
            self.add_pair(pair, -priority + adjustment)
        else:
            self.add_pair(pair, adjustment)
    
    def pop_pair(self):
        'Remove and return the lowest priority pair. Raise KeyError if empty.'
        while self.pq:
            _, _, pair = heappop(self.pq)
            if pair is not self.REMOVED:
                del self.entry_finder[pair]
                return pair
        raise KeyError('pop from an empty priority queue')


def _get_most_frequent_pair(byte_pair_frequencies):
    return byte_pair_frequencies.pop_pair()
    

def _apply_all_merges(
    words: set[tuple[bytes,...]],
    pair: tuple[bytes],
    merged_pair: bytes,
) -> dict[tuple[bytes, ...], tuple[bytes, ...]]:
    """Given a set of words where each contains a merged pair, apply
    all merges, and mapping from new_tokens -> old_tokens where
    old_tokens is the word pre-merge, and new_tokens is the word with the
    merge applied.
    """
    replacements = {}
    for old_tokens in words:
        i, new_tokens = 0, []
        while i < len(old_tokens):
            if i < len(old_tokens)-1 and (old_tokens[i], old_tokens[i+1]) == pair:
                new_tokens.append(merged_pair) 
                i += 2
            else:
                new_tokens.append(old_tokens[i])
                i += 1
        new_tokens = tuple(new_tokens)
        replacements[new_tokens] = old_tokens
    return replacements

def _update_state(
    most_frequent_pair: tuple[bytes],
    merged_pair: bytes,
    byte_frequencies: dict[tuple[bytes], int],
    byte_pair_frequencies: UpdateHeap,
    byte_pair_to_words: dict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]],
) -> tuple[
    dict[tuple[bytes], int],
    UpdateHeap,
    dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
]:
    """Applies a merge to all words containing `most_frequent_pair`, updating and
    returning byte_frequencies, byte_pair_frequencies, and byte_pair_to_words accordingly.
    """
    # Obtain a mapping from words with the merges applied to the word that they replace
    words = byte_pair_to_words[most_frequent_pair]
    replacements = _apply_all_merges(words, most_frequent_pair, merged_pair)
    
    # Update byte_frequencies (word counts) with merge applied
    for new_tokens, old_tokens in replacements.items():
        byte_frequencies[new_tokens] = byte_frequencies[old_tokens]

    # Update the frequency of each byte_pair based on how many were added/removed by the merges
    # Also update the set of words to which each pair maps
    pair_to_adjustment = {}
    added_pair_to_words = defaultdict(Counter)
    for new_tokens, old_tokens in replacements.items():
        old_tokens_pairs_set = set([tuple(pair) for pair in zip(old_tokens[:-1], old_tokens[1:])])
        new_word_pairs = ([tuple(pair) for pair in zip(new_tokens[:-1], new_tokens[1:])])
        new_word_pairs_to_counter = Counter(new_word_pairs)
        new_word_pairs_set = set(new_word_pairs)

        # Calculate adjustments for affected, existing pairs
        for old_pair in old_tokens_pairs_set:
            old_pair_words = byte_pair_to_words[old_pair]
            adjustment = 0
            if old_tokens in old_pair_words:
                adjustment += -(byte_frequencies[old_tokens] * old_pair_words[old_tokens])
                del old_pair_words[old_tokens]
            if old_pair in new_word_pairs_set:
                appearances = new_word_pairs_to_counter[old_pair]
                old_pair_words[new_tokens] = old_pair_words[new_tokens] + appearances
                adjustment += byte_frequencies[new_tokens] * appearances

            pair_to_adjustment[old_pair] = pair_to_adjustment.get(old_pair, 0) + adjustment

            byte_pair_to_words[old_pair] = old_pair_words  

        # Calculate adjustments for newly created pairs
        merged_pair_indices = [i for i in range(len(new_tokens)) if new_tokens[i] == merged_pair]
        added_pairs = set()
        for index in merged_pair_indices:
            if index > 0:
                added_pairs.add((new_tokens[index-1], new_tokens[index]))
            if index < len(new_tokens)-1:
                added_pairs.add((new_tokens[index], new_tokens[index+1]))
        for ap in added_pairs:
            frequency = new_word_pairs_to_counter[ap]
            added_pair_to_words[ap][new_tokens] = frequency
            pair_to_adjustment[ap] = pair_to_adjustment.get(ap, 0) + byte_frequencies[new_tokens] * frequency   
    
    byte_pair_to_words.update(added_pair_to_words)
    for pair, adjustment in pair_to_adjustment.items():
        if adjustment:
            byte_pair_frequencies.adjust_pair_priority(pair, adjustment)

    for old_tokens in replacements.values():
        del byte_frequencies[old_tokens]

    return byte_frequencies, byte_pair_frequencies, byte_pair_to_words
    

def _create_initial_vocab(
    special_tokens: list[str]
) -> tuple[dict[int, bytes], int]:
    """Given a list of special tokens, create an initial mapping from vocab
    indices to byte sequences for each special tokens. Returns the dictionary
    itself and the number of initial entries.
    """
    num_special_tokens = len(special_tokens)
    vocab = { 
        index: bytes([index-num_special_tokens])
        for index in range(num_special_tokens, 256+num_special_tokens)
    }

    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode('utf-8')

    return vocab, len(vocab)

def _get_chunk_word_frequencies(
    input_path: str,
    start_end_pair: tuple[int, int],
    regex_pattern: str,
    special_tokens: list[str],
) -> Counter[str]:
    """Given a file chunk, a regex pattern, and list of special tokens for splitting,
    complete the pre-tokenization word-frequency table construction for the chunk.
    Call this on separate chunks of a file using dedicated processes to do pre-
    tokenization in parallel
    """
    start, end = start_end_pair
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        file_chunk_split_on_special_characters = re.split(
            ('|'.join([re.escape(st) for st in special_tokens])),
            chunk,
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

def _create_byte_pair_frequencies(
    word_frequencies: Counter[str]
) -> tuple[UpdateHeap, dict[tuple[bytes, bytes], Counter[tuple[bytes]]]]:
    byte_pair_frequencies_dict = {}
    byte_pair_to_words = {}

    for word in word_frequencies:
        word_bytes = word.encode('utf-8')
        count = word_frequencies[word]
        for pair in zip(word_bytes[:-1], word_bytes[1:]):
            byte_1, byte_2 = pair
            # Cast back to bytes since list indexing casts to integers
            byte_1, byte_2 = byte_1.to_bytes(), byte_2.to_bytes()  
            byte_pair_frequencies_dict[pair] = byte_pair_frequencies_dict.get(pair, 0) + count
            byte_pair_words = byte_pair_to_words.get((byte_1, byte_2), Counter())
            word_bytes_tuple = (tuple([b.to_bytes() for b in word_bytes]))
            byte_pair_words[word_bytes_tuple] += 1
            byte_pair_to_words[(byte_1, byte_2)] = byte_pair_words

    byte_pair_frequencies = UpdateHeap([
        ((b1.to_bytes(), b2.to_bytes()), count) for (b1, b2), count in
        list(byte_pair_frequencies_dict.items())
    ])

    return byte_pair_frequencies, byte_pair_to_words



def _create_byte_frequencies(
    input_path: str,
    regex_pattern: str,
    special_tokens: list[str],
    num_processes: int,
) -> dict[tuple[bytes], int]:
    """Given a raw corpus and a list of special tokens, splits the corpus by the special tokens
    and a regex pattern, then constructs a mapping from tuples of bytes in the corpus to the 
    frequency at which they appear.
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_CORPUS_CHUNKS, special_tokens[0].encode('utf-8'))

    start_end_pairs = list(zip(boundaries[:-1], boundaries[1:]))
    num_chunks = len(start_end_pairs)
    chunk_input_paths =[input_path] * num_chunks
    chunk_regex_patterns = [regex_pattern] * num_chunks
    chunk_special_tokens = [special_tokens] * num_chunks

    # Pre-tokenize (obtain word counts) each chunk in parallel
    with ProcessPoolExecutor(num_processes) as executor:
        word_frequencies_per_segment_per_chunk = list(
            executor.map(
                _get_chunk_word_frequencies,
                chunk_input_paths,
                start_end_pairs,
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

    byte_pair_frequencies, byte_pair_to_words = _create_byte_pair_frequencies(word_frequencies)

    return byte_frequencies, byte_pair_frequencies, byte_pair_to_words


def train_bpe(
    input_path,
    vocab_size,
    special_tokens: list[str]=['<|endoftext|>']
) -> tuple[dict[int, bytes], list[tuple[tuple[bytes]]]]:
    vocab, initial_vocab_size = _create_initial_vocab(special_tokens)

    byte_frequencies, byte_pair_frequencies, byte_pair_to_words = _create_byte_frequencies(
        input_path=input_path,
        regex_pattern=PAT,
        special_tokens=special_tokens,
        num_processes=NUM_PROCESSES,
    )

    merges = []
    num_merges = len(merges)

    progress_bar = tqdm(total=vocab_size, desc=f"Merges (out of expected {vocab_size})")

    while True:
        most_frequent_pair = _get_most_frequent_pair(byte_pair_frequencies)

        if len(most_frequent_pair) == 0:
            break
        
        merges.append(most_frequent_pair)
        merged_pair = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[initial_vocab_size + num_merges] = merged_pair

        num_merges += 1
        
        byte_frequencies, byte_pair_frequencies, byte_pair_to_words = _update_state(
            most_frequent_pair=most_frequent_pair,
            byte_frequencies=byte_frequencies,
            merged_pair=merged_pair,
            byte_pair_frequencies=byte_pair_frequencies,
            byte_pair_to_words=byte_pair_to_words,
        )
        if len(vocab) == vocab_size:
            break

        progress_bar.update()
    
    progress_bar.close()

    return (vocab, merges)