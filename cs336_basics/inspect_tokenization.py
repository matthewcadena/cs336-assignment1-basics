import json

from cs336_basics import tokenizer

def inspect_compression_ratio():
    owt_tokenizer = tokenizer.Tokenizer.from_files(
        './train_bpe_open_web_train_vocab.json',
        './train_bpe_open_web_train_merges.txt',
    )
    ts_tokenizer = tokenizer.Tokenizer.from_files(
        './train_bpe_tiny_stories_train_vocab.json',
        './train_bpe_tiny_stories_train_merges.txt',
    )
    with open('./data/owt_sample.txt', 'r') as owt, open('./data/TinyStories_sample.txt', 'r') as ts:
        print('Open Web Text tokenization:\n')
        owt_content = owt.read()
        num_owt_bytes = len(owt_content.encode('utf-8', errors='ignore'))
        print(f'Number of bytes in OWT sample: {num_owt_bytes}')
        num_owt_tokens = len(owt_tokenizer.encode(owt_content))
        print(f'Number of tokens in OWT tokenizer encoding: {num_owt_tokens}')
        print(f'OWT compression ratio: {num_owt_bytes / num_owt_tokens}')
        print('----------')

        print('TinyStories tokenization:\n')
        ts_content = ts.read()
        num_ts_bytes = len(ts_content.encode('utf-8', errors='ignore'))
        print(f'Number of bytes in TinyStories sample: {num_ts_bytes}')
        num_ts_tokens = len(ts_tokenizer.encode(ts_content))
        print(f'Number of tokens in TinyStories tokenizer encoding: {num_ts_tokens}')
        print(f'TinyStories compression ratio: {num_ts_bytes / num_ts_tokens}')
        print('----------')

        print('Cross-tokenization 😬')

        ts_encoding_of_owt = ts_tokenizer.encode(owt_content)
        ratio = num_owt_bytes / len(ts_encoding_of_owt)
        print(f'Compression ratio when encoding the OWT sample with the TinyStories encoder: {ratio}')

        owt_encoding_of_ts = owt_tokenizer.encode(ts_content)
        ratio = num_ts_bytes / len(owt_encoding_of_ts)
        print(f'Compression ratio when encoding the TS sample with the OWT encoder: {ratio}')








if __name__ == "__main__":
    inspect_compression_ratio()