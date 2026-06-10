import json

def log_vocab_details(num_tokens: int):
    with open('./train_bpe_open_web_train_vocab.json', 'r') as v:
        vocab = json.load(v)
        
        longest_tokens = sorted(
            vocab.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )[:num_tokens]

        print(f"{num_tokens} longest vocabulary entries:\n")
        for token, index in longest_tokens:
            print(f'{index} : {token} (length {len(token)})')
    
    return

if __name__ == "__main__":
    log_vocab_details(30)