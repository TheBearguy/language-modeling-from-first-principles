# compute_char_freq()
# get_pair_counts()


from statistics import median_grouped


def get_corpus(text):
    """
    takes in the text, 
    returns the corpus with which we play ahead
    """
    words = text.split(" ")
    corpus = []
    for word in words: 
        char_list = list(word)
        char_list.append('</w>')
        corpus.append(char_list)
    return corpus


def get_pair_counts(corpus): 
    """
    takes in the corpus, 
    returns the dict, where key = <pair1, pair2>; value = frequency of the pair
    """
    # print(f"\nCORPUS in the method :: {corpus}\n")
    counts = {}
    for word in corpus: 
        # print(f"\nWord in the method: {word}\n")
        for i in range(len(word)-1): 
            pair = (word[i], word[i+1]) 
            counts[pair] = counts.get(pair, 0) + 1
            # print(f"Counts :: {counts}")
    return counts


def merge_pairs(corpus, pair_freq): 
    """
    takes in the corpus and the dict of pairs and their frequency, 
    returns the updated corpus
    """

    pair_to_merge = max(pair_freq, key=pair_freq.get) 
    new_corpus = []
    for word in corpus: 
        new_word = []
        i = 0
        while i < len(word): 
            if i < len(word) - 1 and (word[i], word[i+1]) == pair_to_merge: 
                merged_token = word[i] + word[i+1]
                new_word.append(merged_token)
                i += 2
            else: 
                new_word.append(word[i])
                i += 1
        new_corpus.append(new_word)
    return new_corpus, pair_to_merge 

# Training learns merge rules from the given training data
def train(text, target = 100): 
    corpus = get_corpus(text)
    # Building the vocab - init stage
    merge_rules = []
    vocab = set()
    for word in corpus: 
        for char in word: 
            vocab.add(char)

    print(f"\nInitial Vocab size : {len(vocab)}")

    while len(vocab) < target: 
        pair_freq = get_pair_counts(corpus)
        if not pair_freq: 
            break
        corpus, pair_to_merge = merge_pairs(corpus, pair_freq)

        # Add merged token to vocab
        merged_token = pair_to_merge[0] + pair_to_merge[1]
        vocab.add(merged_token)
        merge_rules.append(pair_to_merge)

        print(f"Merged {pair_to_merge} → vocab size: {len(vocab)}")
    
    print(f"\nFinal vocab :: {sorted(vocab)}")
    print(f"\nFinal Corpus :: {corpus}")    

    vocab_to_id = {token : idx for idx, token in enumerate(sorted(vocab))}
    return vocab, merge_rules, vocab_to_id, corpus

# Encoding, applies the learned merge_rules to the text
def encode(text, merge_rules, vocab_to_id): 
    """
    Encode text into token IDs, using learned merge rules
    """
    corpus = get_corpus(text)

    for pair_to_merge in merge_rules: 
        new_corpus = []
        for word in corpus: 
            new_word = []
            i = 0
            while i < len(word): 
                if i < len(word) -1 and (word[i], word[i+1]) == pair_to_merge: 
                    merged_token = word[i] + word[i+1]
                    new_word.append(merged_token)
                    i += 2
                else: 
                    new_word.append(word[i])
                    i += 1
            new_corpus.append(new_word)
        corpus = new_corpus

    token_ids = []
    for word in corpus: 
        for token in word: 
            if token in vocab_to_id: 
                token_ids.append(vocab_to_id[token])
            else: 
                #Handle unknown tokens
                token_ids.append(vocab_to_id.get("<UNK>", 0))
    return token_ids, corpus




if __name__ == "__main__": 
    target = 100
    text = "The penguin started heading towards the mountains; some 70 kms away"
    # Train once: 
    voca, merge_rules, vocab_to_id, corpus = train(text, target)

    print(f"\nEncoding NEW TEXT\n")

    # Now encode new text using the learned rules: 
    new_text = "penguin heading towards the mountains, for a purpose"
    token_ids, encoded_corpus = encode(new_text, merge_rules, vocab_to_id)

    print(f"\nNew text: {new_text}\n")
    print(f"\nEncoded tokens: {encoded_corpus}\n")
    print(f"\nToken IDs: {token_ids}\n")