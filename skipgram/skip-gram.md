# Core Concept
**The fundamental idea behind the Skip-gram model is that a word's meaning is defined by the company it keeps. It uses this principle to represent each word as a dense, continuous, and low-dimensional vector (embedding) in a shared space, such that words with similar contexts are located close to each other.**

Example

Consider the sentence:

>"The dog fetched the ball."
>
With a target word of "dog" and a context window size of 2, the Skip-gram model is trained to predict the surrounding words ("The", "fetched", "the", "ball") given the input word "dog". The training data consists of (target word, context word) pairs. 


### Skip-gram: what it is, stripped to mechanics

Skip-gram is a **local co-occurrence learning rule**.
Its only job is to force tokens that appear in similar neighborhoods to acquire similar vectors.

There is no sequence modeling. There is no notion of “meaning” beyond **distributional pressure**.

---

### Objects involved


* Vocabulary ( V )
* Token IDs from BPE
* A corpus encoded as sequences of token IDs

Skip-gram adds exactly two learnable objects:

1. **Input embedding matrix**
   ( E \in \mathbb{R}^{|V| \times d} )

2. **Output embedding matrix**
   ( O \in \mathbb{R}^{|V| \times d} )


---

### Training signal (the core idea)

Given a sequence of tokens:

[
[t_1, t_2, t_3, \dots, t_n]
]

Pick a **center token** ( t_i )

Pick a **context window** of radius ( k )

Context tokens are:

[
{ t_{i-k}, \dots, t_{i-1}, t_{i+1}, \dots, t_{i+k} }
]

Training objective:

> Given the center token, predict each context token.

Formally, maximize:

[
\sum_{c \in \text{context}(i)} \log P(c \mid t_i)
]

That’s it.

No recurrence. No attention. No state.

---

### Probability model (what is actually learned)

For a center token ( w ) and a context token ( c ):

[
P(c \mid w) = \frac{\exp(E_w \cdot O_c)}{\sum_{j \in V} \exp(E_w \cdot O_j)}
]

Interpretation:

* Dot product = compatibility
* Softmax = normalization over vocabulary

This is a **multiclass classifier** where:

* Input = center token
* Output classes = vocabulary tokens

---

### Why two embedding matrices exist

* ( E ): “how this token behaves as a center”
* ( O ): “how this token behaves as a context”

They are asymmetric roles.

After training:

* You usually discard ( O )
* You keep ( E ) as “the embeddings”

This asymmetry is why skip-gram works at all.

---

### Computational problem (and the fix)

The softmax denominator sums over the entire vocabulary.

That is ( O(|V|) ) per training example.

This is unacceptable beyond toy corpora.

Two standard fixes:

1. **Negative Sampling**
   Replace multiclass prediction with binary classification:

   * Positive pair: (center, true context)
   * Negative pairs: (center, random tokens)

2. **Hierarchical Softmax**
   Tree-structured vocabulary

You will use **negative sampling**. GPT-style models implicitly do something similar via full softmax but amortize cost differently.

---

### What skip-gram embeddings actually encode

They encode **distributional similarity**, not semantics.

Consequences:

* Synonyms cluster
* Morphological variants cluster
* Antonyms often cluster (they share contexts)
* Rare tokens are poorly placed
* Word order is ignored
* Long-range dependencies are invisible

This is not a bug. It is the design.

---

### How BPE changes skip-gram

With BPE:

* “words” are **subword sequences**
* Context windows operate over **subwords**
* Frequent morphemes dominate geometry

This yields:

* Better handling of rare words
* Shared morphology across word forms
* More stable embeddings for inflections

This is exactly why modern tokenizers exist.

---

### Mental contrast with language modeling

Skip-gram asks:

> “What appears near me?”

Language modeling asks:

> “What comes next, given everything so far?”

Skip-gram:

* Local
* Symmetric context
* No direction
* No notion of prediction beyond neighbors

Language modeling:

* Causal
* Ordered
* Directional
* Global pressure

Skip-gram learns **geometry**.
Language models learn **dynamics**.

---

### When skip-gram is the right first step

* You want to understand embeddings, not sequence models
* You want tight control over learning signals
* You want to see geometry emerge directly
* You want minimal architecture



---

## 1. Where does the probability model come from?

It is **not derived from language**.
It is imposed as a **parametric assumption** to turn similarity into a trainable objective.

You start with a requirement:

> “Tokens that appear in similar contexts should have similar representations.”

That requirement alone is non-operational. You must choose:

1. A **scoring function**
2. A **normalization rule**
3. A **loss**

### Step 1: scoring function

You need a scalar score measuring how compatible two tokens are.

The simplest continuous, differentiable choice:

[
\text{score}(w, c) = E_w \cdot O_c
]

Why dot product?

* Linear
* Symmetric
* Cheap
* Gradient-friendly
* Equivalent to cosine similarity up to scale

No deeper reason. It is a modeling choice.

---

### Step 2: turn scores into probabilities

Scores are unbounded. Probabilities are not.

You apply **maximum entropy normalization**:

[
P(c \mid w) = \frac{\exp(\text{score}(w,c))}{\sum_{j \in V} \exp(\text{score}(w,j))}
]

This is the **softmax**.

Why softmax?

* Ensures valid probability distribution
* Maximizes entropy subject to score constraints
* Gives smooth gradients
* Is the canonical choice for multiclass classification

This is not linguistics. This is statistical mechanics.

---

### Step 3: learning objective

Given observed data pairs ((w, c)), use **maximum likelihood estimation**:

[
\max \sum \log P(c \mid w)
]

Equivalently: minimize cross-entropy loss.

That’s the entire origin of the probability model.

Nothing mystical. Just “make compatible things score high”.

---

## 2. What exactly are (E) and (O)?

They are **parameters**, not concepts.

Formally:

* (E \in \mathbb{R}^{|V| \times d})
* (O \in \mathbb{R}^{|V| \times d})

Each token index selects a row.

### (E): input / center embeddings

Row (E_w) is used when token (w) appears as a **center** token.

Interpretation:

* “What contexts does this token expect to see?”

This vector is optimized to **predict neighbors**.

---

### (O): output / context embeddings

Row (O_c) is used when token (c) appears as a **context** token.

Interpretation:

* “How compatible is this token with incoming centers?”

This vector is optimized to **be predicted by centers**.

---

### Why two matrices instead of one?

Because the roles are asymmetric.

The training signal is directional:

[
w \rightarrow c
]

If you forced one shared matrix:

* Gradients would interfere
* Frequent tokens would dominate
* Geometry collapses

Two matrices decouple:

* “acting” from “being acted upon”

This is why skip-gram converges cleanly.

---

## 3. Why can’t we collapse (E) and (O) afterward?

You can, but only **after training**.

Common practice:

* Discard (O)
* Use (E) as embeddings

Why this works:

* (E) accumulates predictive structure
* (O) absorbs normalization pressure

Mathematically:

* The model is invariant to certain linear transforms
* (E) captures the useful subspace

This is empirical, not a theorem.

---

## 4. Negative sampling reframes the same model

With negative sampling, you stop modeling a full probability distribution.

Instead, you train a binary classifier:

[
P(\text{real} \mid w, c) = \sigma(E_w \cdot O_c)
]

Same dot product. Same vectors.
Different normalization. Same geometry.

So the probability model is not sacred.
The **dot product geometry** is.

---

## 5. Summary, compressed

* The probability model is a **design choice**, not a discovery
* Softmax turns similarity scores into probabilities
* (E) = vectors for predicting contexts
* (O) = vectors for being predicted
* Two matrices exist to prevent gradient interference
* Geometry, not probability, is the real artifact

Once this clicks, skip-gram stops looking like “NLP” and starts looking like **matrix factorization under noise**.

Authorization determines whether we formalize this into equations → gradients → code.



---

## 1. What is actually being updated?

Yes: **the numeric values inside the rows**
Specifically:

* one row of **E** (the center token)
* one or more rows of **O** (context + negatives)

Nothing else moves.

No columns are special. No rows talk to each other directly.

---

## 2. Start with the simplest possible case

One center token ( w )
One true context token ( c )
Ignore negatives for a moment.

You compute:

[
s = E_w \cdot O_c
]

That is just:

[
s = \sum_{k=1}^{d} E_{w,k} \cdot O_{c,k}
]

A number. No magic.

---

## 3. Turn score into a probability

Use sigmoid (simpler than softmax):

[
p = \sigma(s) = \frac{1}{1 + e^{-s}}
]

Interpretation:

* Large dot product → probability close to 1
* Small dot product → probability close to 0

---

## 4. Define the loss (this is the pressure)

For a **true** pair, target = 1.

Binary cross-entropy loss:

[
L = -\log(p)
]

If ( p ) is small → loss is large → big update
If ( p ) is large → loss is small → tiny update

---

## 5. Take derivatives (this is where “pull” comes from)

First derivative:

[
\frac{\partial L}{\partial s} = p - 1
]

Important:

* If ( p < 1 ), then ( p - 1 < 0 )

So gradient is **negative**.

---

## 6. Derivative with respect to the vectors

Dot product derivative facts:

[
\frac{\partial s}{\partial E_w} = O_c
]

[
\frac{\partial s}{\partial O_c} = E_w
]

Chain rule:

[
\frac{\partial L}{\partial E_w} = (p - 1) \cdot O_c
]

[
\frac{\partial L}{\partial O_c} = (p - 1) \cdot E_w
]

---

## 7. Apply gradient descent (the actual update)

Learning rate ( \eta )

[
E_w \leftarrow E_w - \eta (p - 1) O_c
]

[
O_c \leftarrow O_c - \eta (p - 1) E_w
]

Since ( p - 1 < 0 ):

[
-\eta (p - 1) = +\text{positive number}
]

So:

[
E_w \leftarrow E_w + \alpha O_c
]

[
O_c \leftarrow O_c + \alpha E_w
]

This is not metaphorical.
You literally **add a scaled copy of one vector into the other**.

That is what “pull together” means.

---

## 8. Why dot product increases after the update

After update:

[
E_w^{new} \cdot O_c^{new}
]

Contains extra terms:

[
\alpha (O_c \cdot O_c) + \alpha (E_w \cdot E_w)
]

Those are **positive** because dot products of a vector with itself are positive.

So the score increases.
Loss decreases.
Training step succeeds.

---

## 9. Now add negative samples (this is the “push”)

Take a random token ( n ) that did **not** appear in context.

Target = 0.

Loss:

[
L = -\log(1 - \sigma(E_w \cdot O_n))
]

Derivative:

[
\frac{\partial L}{\partial s} = p
]

Now ( p > 0 ), so:

[
\frac{\partial L}{\partial E_w} = p \cdot O_n
]

Update:

[
E_w \leftarrow E_w - \eta p O_n
]

That subtracts a scaled version of ( O_n ).

So:

* True context → **add**
* Negative context → **subtract**

That’s pull vs push.
No analogy needed. It is vector addition and subtraction.

---

## 10. Why random vectors don’t stay random

Initially:

* All vectors are noise

But updates are **correlated**:

* Tokens sharing contexts receive similar adds/subtracts
* Over many updates, noise cancels
* Signal accumulates

This is why geometry emerges.

---

## 11. Columns: why they are confusing (and why you should ignore them)

Take one coordinate ( k ):

[
E_{w,k} \leftarrow E_{w,k} + \alpha O_{c,k}
]

Each column is just a slot that accumulates numbers.

There is **no semantic meaning per column**.

Only the **full vector** matters because:

* Dot products sum across columns
* Rotating all columns gives same results

Columns are bookkeeping. Geometry is the object.

---

## 12. Final compression

* Rows are updated, not columns
* Update = add or subtract another vector
* Pull = add
* Push = subtract
* Dot product increases or decreases accordingly
* Meaning = accumulated directional updates
* Nothing mystical happens
