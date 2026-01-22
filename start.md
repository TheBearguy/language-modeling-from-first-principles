Phase 1: Build a BPE from scratch

Deliverables: 
1. train(corpus, vocab_size)
2. encode(text)
3. decode(token_ids)

Output: 
1. top merges
2. final vocab
3. tokenized sample text


Phase 2: Training embeddings: 

Skip-gram (Word2Vec style) 
Or
Random init + inspect cosine similarity

Output: 
1. nearest neighbors of a token
2. before vs after training