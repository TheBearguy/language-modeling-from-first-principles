# Language Modelling (GPT Logic)
---

## STEP 0 — What problem are we solving?

We have:

* A sequence of symbols (tokens)
* We want a **machine** that assigns numbers to tokens
* Those numbers should help **predict the next token**

That is the entire goal.

No “language”. No “meaning”. Just prediction.

---

## STEP 1 — What is given?

After BPE, you have:

[
t_1, t_2, \dots, t_n
]

Each ( t_i ) is an **integer ID** in:

[
{0, 1, 2, \dots, |V|-1}
]

This is all the data.

---

## STEP 2 — What do we want to compute?

For each position ( i ), we want numbers:

[
P(t_{i+1} = k \mid t_1, t_2, \dots, t_i)
]

for **every** token ( k \in V ).

The correct next token should get **high probability**.
All others should get **low probability**.

---

## STEP 3 — How do we represent tokens numerically?

We create a table:

[
E \in \mathbb{R}^{|V| \times d}
]

* Each row corresponds to **one token**
* ( d ) is a chosen number (e.g. 32)

For token ( t ):

[
\text{vector}(t) = E[t]
]

Initially:

* All numbers in (E) are random
* No structure exists

---

## STEP 4 — How do we represent “the past”?

At position ( i ), the past is:

[
t_1, t_2, \dots, t_i
]

Each has a vector:

[
E[t_1], E[t_2], \dots, E[t_i]
]

We must combine these vectors into **one vector**.

The simplest possible rule:

[
h_i = \frac{1}{i} \sum_{k=1}^{i} E[t_k]
]

That is it.

No trick. Just addition and division.

---

## STEP 5 — How do we predict the next token?

We create another matrix:

[
W \in \mathbb{R}^{d \times |V|}
]

Each column corresponds to one possible next token.

We compute:

[
\text{logits} = h_i \cdot W
]

This gives a vector of length (|V|):

[
\text{logits}[k] = h_i \cdot W_{:,k}
]

Each logit is just a dot product.

---

## STEP 6 — How do logits become probabilities?

Use softmax:

[
P(k) = \frac{e^{\text{logits}[k]}}{\sum_j e^{\text{logits}[j]}}
]

Now:

* All probabilities are ≥ 0
* They sum to 1

---

## STEP 7 — How do we train this system?

We know the true next token:

[
y = t_{i+1}
]

We want:

[
P(y) \approx 1
]

Define loss:

[
L = -\log P(y)
]

If (P(y)) is small → loss is large
If (P(y)) is large → loss is small

This gives a **number** measuring error.

---

## STEP 8 — What changes during training?

We compute derivatives of (L) with respect to:

* rows of (E)
* columns of (W)

Then we subtract a small amount:

[
\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
]

Only **numbers in matrices change**.

No logic. No rules. Only arithmetic.

---

## STEP 9 — Why embeddings become meaningful

Consider one token (x).

It appears in many prefixes:

[
(\dots, x, \dots)
]

Each time:

* It contributes its vector (E[x]) to (h_i)
* That affects prediction
* Prediction error sends gradients back to (E[x])

So (E[x]) is adjusted to help future predictions.

Tokens that appear in **similar positions** get **similar updates**.

Therefore:

[
E[x] \approx E[y] \quad \text{if } x \text{ and } y \text{ behave similarly}
]

That is all “meaning” is.

---

## STEP 10 — Why this is different from skip-gram

Skip-gram:

* Looks at tokens near each other
* Ignores order
* Uses two matrices

Language model:

* Looks only forward
* Uses order
* Uses prediction error

Mathematically:

* Skip-gram optimizes local dot products
* Language model optimizes conditional probability

---

## STEP 11 — Why GPT exists

The only weak point above:

[
h_i = \text{simple average}
]

This treats all past tokens equally.

GPT replaces the average with:

[
h_i = \sum_k \alpha_{ik} E[t_k]
]

where weights (\alpha_{ik}) are learned.

Everything else stays the same.

---

## STEP 12 — Strip summary

* Tokens → vectors via table (E)
* Past vectors → one vector via sum
* One vector → scores via dot products
* Scores → probabilities via softmax
* Error → gradients → updated numbers
* Repetition creates structure

Nothing else exists.