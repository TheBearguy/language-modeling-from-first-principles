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



# Explaining the code (coz it seemed hard to me)

## The Big Picture

This code builds a **simple neural language model** that learns to predict the next token in a sequence. Think of it like autocomplete on your phone—given some words, what word comes next?

---

## Code Walkthrough

### 1. **`init_lm_params` - Setting up the neural network**

```python
E = rng.normal(0, 0.01, (vocab_size, dim))  # token embeddings
W = rng.normal(0, 0.01, (dim, vocab_size))  # output projections
```

**What's happening:**
- **E (embeddings matrix)**: Each token (word/subword) gets a vector of numbers. If you have 1000 tokens and `dim=32`, you get a 1000×32 matrix. Each row is a token's "meaning" in 32 dimensions.
- **W (weights matrix)**: This projects from the hidden dimension back to vocabulary space (32×1000). It's used to score which token should come next.

**Intuition**: E converts tokens to vectors, W converts vectors back to token predictions.

---

### 2. **`softmax` - Converting scores to probabilities**

```python
def softmax(x): 
    x = x - np.max(x)  # numerical stability trick
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)
```

Takes raw scores (logits) and converts them to probabilities that sum to 1.

Example: `[2.0, 1.0, 0.1]` → `[0.659, 0.242, 0.099]`

---

### 3. **`lm_steps` - The core training step**

This is where the learning happens. Let me break it into parts:

#### **Forward Pass (Making a prediction)**

```python
h = E[prefix_ids].mean(axis=0)
```
- Takes all prefix tokens (e.g., "the cat sat on")
- Looks up their embeddings from E
- **Averages them** to get a single context vector `h`

This is a **bag-of-words** approach—simple but loses word order.

```python
logits = np.dot(h, W)
probs = softmax(logits)
```
- Projects `h` through W to get scores for every possible next token
- Converts to probabilities

```python
loss = -np.log(probs[target_id])
```
- Calculates **cross-entropy loss**: "How wrong were we?"
- If the model predicted 90% for the correct token, loss is low
- If it predicted 1%, loss is high

#### **Backward Pass (Learning from mistakes)**

```python
dlogits = probs
dlogits[target_id] -= 1
```
- This is the gradient of cross-entropy + softmax (combined)
- For the correct token: `prob - 1` (negative if prob < 1)
- For wrong tokens: just `prob` (positive)

**Intuition**: Push down wrong tokens, push up the right token.

```python
dw = np.outer(h, dlogits)
W = W - lr*dw
```
- Update W to make it better at predicting this example
- `lr` (learning rate) controls step size

```python
dh = np.dot(W, dlogits)
gradient_per_token = dh/len(prefix_ids)
for t in prefix_ids: 
    E[t] = E[t] - lr * gradient_per_token
```
- Backpropagate error to embeddings
- Since we averaged prefix tokens, **split the gradient evenly** among them
- Update each prefix token's embedding

---

### 4. **`train_language_model` - The training loop**

```python
for i in range(1, len(token_ids)):
    prefix = token_ids[:i]
    target = token_ids[i]
```

**Example**: If `token_ids = [42, 17, 8, 99, 3]`

- Step 1: prefix=`[42]`, target=`17`
- Step 2: prefix=`[42, 17]`, target=`8`  
- Step 3: prefix=`[42, 17, 8]`, target=`99`
- Step 4: prefix=`[42, 17, 8, 99]`, target=`3`

The model learns to predict each token based on everything before it.

---

## Key Architectural Choices

1. **Bag-of-words averaging**: Simple but ignores order. Modern models use attention mechanisms instead.
2. **No positional encoding**: The model can't tell "cat sat" from "sat cat".
3. **Direct SGD**: Updates after every example (vs. batching).
4. **Tiny architecture**: No hidden layers, just embeddings → linear → softmax.

---

## What's Missing from Production LMs

This is a **teaching example**. Real models (GPT, etc.) add:
- Attention mechanisms (not averaging)
- Positional encodings
- Multiple layers
- Layer normalization
- Better optimizers (Adam)
- Batching
- Dropout/regularization

But the **core idea is the same**: learn embeddings and weights that predict the next token well.

# Backward Pass - Step by Step

## Setup

You have:
- `h` = context vector (size: `dim`, e.g., 32)
- `W` = weight matrix (size: `dim × vocab_size`, e.g., 32×1000)
- `probs` = probability distribution over vocab (size: `vocab_size`, e.g., 1000)
- `target_id` = the correct token (e.g., 542)

---

## Step 1: Compute gradient of loss w.r.t. logits

```python
dlogits = probs
dlogits[target_id] -= 1
```

**Math:**
- Loss = `-log(probs[target_id])`
- When you take derivative of cross-entropy loss + softmax together, you get:
  - `d(loss)/d(logits[i])` = `probs[i]` if `i ≠ target_id`
  - `d(loss)/d(logits[target_id])` = `probs[target_id] - 1`

**Example:**
```
probs = [0.1, 0.7, 0.05, 0.15]
target_id = 1

dlogits = [0.1, 0.7, 0.05, 0.15]  # copy probs
dlogits[1] -= 1                     # subtract 1 from target
dlogits = [0.1, -0.3, 0.05, 0.15]  # result
```

**Intuition:**
- Correct token (index 1): gradient is **negative** (-0.3) → increase its logit
- Wrong tokens: gradients are **positive** → decrease their logits

---

## Step 2: Compute gradient w.r.t. W

```python
dw = np.outer(h, dlogits)
```

**Forward pass was:**
```
logits = h @ W
```
Where `@` means matrix multiply: `logits[j] = Σᵢ h[i] * W[i,j]`

**Backward pass:**
```
dW[i,j] = d(loss)/d(W[i,j])
        = d(loss)/d(logits[j]) * d(logits[j])/d(W[i,j])
        = dlogits[j] * h[i]
```

**Outer product creates this:**
```python
dW = [[h[0]*dlogits[0], h[0]*dlogits[1], ..., h[0]*dlogits[vocab_size-1]],
      [h[1]*dlogits[0], h[1]*dlogits[1], ..., h[1]*dlogits[vocab_size-1]],
      ...
      [h[dim-1]*dlogits[0], ..., h[dim-1]*dlogits[vocab_size-1]]]
```

**Shape:** `(dim, vocab_size)` - same as W

**Concrete example:**
```
h = [0.5, 0.3]  (dim=2)
dlogits = [0.1, -0.3, 0.2]  (vocab_size=3)

dW = [[0.5*0.1,  0.5*(-0.3),  0.5*0.2 ],
      [0.3*0.1,  0.3*(-0.3),  0.3*0.2 ]]

   = [[0.05, -0.15, 0.10],
      [0.03, -0.09, 0.06]]
```

---

## Step 3: Compute gradient w.r.t. h

```python
dh = np.dot(W, dlogits)
```

**Forward was:**
```
logits = h @ W  
# which is: logits[j] = Σᵢ h[i] * W[i,j]
```

**Backward (chain rule):**
```
dh[i] = Σⱼ d(loss)/d(logits[j]) * d(logits[j])/d(h[i])
      = Σⱼ dlogits[j] * W[i,j]
      = W[i,:] @ dlogits
```

**This is matrix-vector multiply:**
```python
dh = W @ dlogits
```

**Shape:** `(dim,)` - same as h

**Concrete example:**
```
W = [[0.2, 0.4, 0.1],
     [0.3, 0.5, 0.2]]  (2×3)

dlogits = [0.1, -0.3, 0.2]

dh[0] = 0.2*0.1 + 0.4*(-0.3) + 0.1*0.2 = 0.02 - 0.12 + 0.02 = -0.08
dh[1] = 0.3*0.1 + 0.5*(-0.3) + 0.2*0.2 = 0.03 - 0.15 + 0.04 = -0.08

dh = [-0.08, -0.08]
```

---

## Step 4: Update W

```python
W = W - lr * dW
```

**Gradient descent:**
- If `dW[i,j]` is positive → decrease `W[i,j]`
- If `dW[i,j]` is negative → increase `W[i,j]`

**Example with lr=0.1:**
```
W_old = [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]

dW = [[0.05, -0.15, 0.10],
      [0.03, -0.09, 0.06]]

W_new = W_old - 0.1 * dW

W_new = [[1.0 - 0.005,  2.0 - (-0.015),  3.0 - 0.010],
         [4.0 - 0.003,  5.0 - (-0.009),  6.0 - 0.006]]

      = [[0.995, 2.015, 2.990],
         [3.997, 5.009, 5.994]]
```

---

## Step 5: Update embeddings E

```python
gradient_per_token = dh / len(prefix_ids)
for t in prefix_ids: 
    E[t] = E[t] - lr * gradient_per_token
```

**Why divide by len(prefix_ids)?**

Remember forward pass:
```python
h = E[prefix_ids].mean(axis=0)
h = (E[t1] + E[t2] + ... + E[tn]) / n
```

**Backward (chain rule):**
```
d(loss)/d(E[ti]) = d(loss)/d(h) * d(h)/d(E[ti])
                 = dh * (1/n)
```

Each token in prefix contributed `1/n` to `h`, so each gets `1/n` of the gradient.

**Concrete example:**
```
prefix_ids = [5, 12, 8]  # 3 tokens
dh = [-0.08, -0.08]
lr = 0.1

gradient_per_token = [-0.08, -0.08] / 3 = [-0.0267, -0.0267]

E[5]  = E[5]  - 0.1 * [-0.0267, -0.0267] = E[5]  + [0.00267, 0.00267]
E[12] = E[12] - 0.1 * [-0.0267, -0.0267] = E[12] + [0.00267, 0.00267]
E[8]  = E[8]  - 0.1 * [-0.0267, -0.0267] = E[8]  + [0.00267, 0.00267]
```

All three prefix tokens get **same gradient** because we averaged them.

---

## Summary in pure logic:

1. `dlogits = probs; dlogits[target] -= 1` → gradient of loss w.r.t. logits
2. `dW = outer(h, dlogits)` → gradient of loss w.r.t. W  
3. `dh = W @ dlogits` → gradient of loss w.r.t. h (backprop through matrix multiply)
4. `W -= lr * dW` → update W via gradient descent
5. `E[each prefix token] -= lr * dh/n` → update embeddings (split gradient equally)