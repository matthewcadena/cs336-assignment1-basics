import json
import regex as re
from typing import Iterable, Iterator, Self
from cs336_basics.common import PAT
from tests.common import gpt2_bytes_to_unicode


class Tokenizer():
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,    
    ):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges

        # split on longest special_token first to avoid splitting on 
        # overlapping delimiters
        special_tokens = [] if special_tokens is None else sorted(
            special_tokens,
            key=lambda x: len(x),
            reverse=True,
        )

        self.special_tokens: list[str] = special_tokens

        for special_token in self.special_tokens:
                byte_encoded_special_token = special_token.encode('utf-8')
                if byte_encoded_special_token not in set(vocab.values()):
                    self.vocab[len(self.vocab)] = byte_encoded_special_token

        self.vocab_item_to_index = { v:k for k,v in self.vocab.items() }
        self.merge_priorities =  { merge : i for i, merge in enumerate(self.merges)}


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        gpt2_byte_decoder = { v:k for k, v in gpt2_bytes_to_unicode().items() }
        with open(vocab_filepath, 'r') as v, open(merges_filepath, 'r') as m:
            raw_vocab = json.load(v)
            vocab = {
                index: bytes([gpt2_byte_decoder[token] for token in vocab_item])
                for vocab_item, index in raw_vocab.items()
            }
            special_tokens = [] if special_tokens is None else special_tokens
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode('utf-8')
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

            text_merges = [tuple(line.rstrip().split(" ")) for line in m]
            gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in text_merges
            ]
        
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        encoding = []

        text_split_on_special_characters = re.split(
            (rf'({'|'.join([re.escape(st) for st in self.special_tokens])})'),
            text,
        ) if self.special_tokens else [text]

        pre_tokenized_segments = []
        for segment in text_split_on_special_characters:
            if segment in self.special_tokens:
                # append special token bytes without pre-tokenizing them 
                pre_tokenized_segments.append(segment.encode('utf-8', errors='replace'))
            else:
                pre_tokenized_segments.append(re.finditer(PAT, segment))


        for segment in pre_tokenized_segments:
            if isinstance(segment, bytes):
                encoding.append(self.vocab_item_to_index[segment])
                continue
            for pre_token in segment:
                pre_token_bytes = pre_token.group().encode('utf-8', errors='replace')
                tokens = [b.to_bytes() for b in pre_token_bytes]
                while True:
                    min_merge, min_priority = None, float('inf')
                    for pair in zip(tokens[:-1], tokens[1:]):
                        if pair in self.merge_priorities and self.merge_priorities[pair] < min_priority:
                            min_merge = pair
                            min_priority = self.merge_priorities[pair]
                    
                    if min_merge is None:
                        break

                    i = 0
                    merged = min_merge[0] + min_merge[1]
                    while i < len(tokens)-1:
                        if (tokens[i], tokens[i+1]) == min_merge:
                            tokens = tokens[:i] + [merged] + tokens[i+2:]
                            i += 1
                        i += 1                   

                for token in tokens:
                    encoding.append(self.vocab_item_to_index[token])
        
        return encoding

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        return (id for line in iterable for id in self.encode(line))

    def decode(
        self,
        ids: list[int],
    ) -> str:
        return b''.join([self.vocab[id] for id in ids]).decode('utf-8', errors='replace')
