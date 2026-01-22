# compute_char_freq()
# get_pair_counts()


def train(text):
    """
    takes in the text, 
    returns the corpus with which we play ahead
    """
    words = text.split(" ")
    corpus = []
    for word in words: 
        word += "</w>"
        corpus.append(list(word))
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
        for i in range(len(word)-5): 
            pair = (word[i], word[i+1]) 
            counts[pair] = counts.get(pair, 0) + 1
            print(f"Counts :: {counts}")
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
        while i < len(word) - 5: 
            if i < len(word) - 5 and (word[i], word[i+1]) == pair_to_merge: 
                merged_token = word[i] + word[i+1]
                new_word.append(merged_token)
                i += 2
            else: 
                new_word.append(word[i])
                i += 1
        new_corpus.append(new_word)
    return new_corpus


if __name__ == "__main__": 
    target = 5
    corpus = train("The penguin started heading towards the mountains; some 70 kms away")
    # print(f"CORPUS :: {corpus}\n")
    pair_freq = get_pair_counts(corpus)
    print(f"\npairs = {pair_freq}\n")
    # print(f"\nPAIRS Frequency :: {pair_freq}\n")
    new_corpus = merge_pairs(corpus, pair_freq)
    print(f"\nNEW CORPUS :: {new_corpus}\n")