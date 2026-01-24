# Recurrent neural networks
---

## 0. The problem RNNs exist to solve

You have a sequence:

[
t_1, t_2, t_3, \dots, t_n
]

At step (i), you want a representation of **everything before** (t_i).

Formally, you want a function:

[
h_i = f(t_1, t_2, \dots, t_i)
]

with these properties:

1. **Order matters**
2. **Past influences future**
3. **Computation grows linearly**, not quadratically
4. **Same rules apply at every step**

Your mean-pooling model failed property (1).
Your positional fix helped, but still failed (2) in a strong way.

---

## 1. Why mean pooling is fundamentally weak

You used:

[
h_i = \frac{1}{i} \sum_{k=1}^{i} x_k
]

where (x_k = E[t_k] + P[k]).

This has two mathematical problems.

### Problem 1: No state

The representation of the prefix is recomputed **from scratch** every time.

There is no memory variable that evolves.

### Problem 2: Equal contribution

Every token contributes with weight (1/i).

There is no way to express:

* “recent tokens matter more”
* “this token is more important than that one”

This is not a training issue.
This is a **representation limitation**.

---

## 2. What RNNs introduce (the single new idea)

RNNs introduce **one persistent variable**:

[
h_i
]

called the **hidden state**.

Instead of recomputing from scratch, you **update** it:

[
h_i = g(h_{i-1}, x_i)
]

That’s the whole idea.

Nothing else.

---

## 3. What “recurrent” actually means

“Recurrent” means:

* The same function (g)
* The same parameters
* Applied repeatedly over time

So the model is **time-invariant**.

Formally:

[
h_1 = g(h_0, x_1)
]
[
h_2 = g(h_1, x_2)
]
[
h_3 = g(h_2, x_3)
]

The same computation repeats.

This is crucial:

* The model does not know where it is in the sequence
* It only knows what state it is in

---

## 4. The simplest possible RNN (exact math)

We choose:

[
g(h, x) = \tanh(W_h h + W_x x)
]

So:

[
h_i = \tanh(W_h h_{i-1} + W_x x_i)
]

Where:

* (h_i \in \mathbb{R}^d)
* (x_i \in \mathbb{R}^d)
* (W_h, W_x \in \mathbb{R}^{d \times d})

Initial state:

[
h_0 = 0
]

That’s it.

---

## 5. Why this solves the problems

### 5.1 Order is intrinsic

If you swap tokens:

[
x_1, x_2 \neq x_2, x_1
]

Because:

[
\tanh(W_h \tanh(W_h h_0 + W_x x_1) + W_x x_2)
\neq
\tanh(W_h \tanh(W_h h_0 + W_x x_2) + W_x x_1)
]

Order changes the result.

No positional hack required.

---

### 5.2 Past influences future non-uniformly

Unroll the recurrence:

[
h_i = \tanh(W_h^i h_0 + \sum_{k=1}^{i} W_h^{i-k} W_x x_k)
]

This equation is key.

It shows:

* Token (x_k) is multiplied by (W_h^{i-k})
* Older tokens are repeatedly transformed
* Recent tokens are transformed fewer times

So **recency bias** emerges naturally.

---

## 6. Why this is a big conceptual upgrade

Mean pooling:

* history = static bag

RNN:

* history = dynamic state

The RNN learns **how information flows forward**.

It does not need to “look back”.
It carries the past inside the present.

---

## 7. How prediction fits in (language modeling)

At each step:

[
P(t_{i+1} \mid t_1, \dots, t_i) = \text{softmax}(h_i W_o)
]

Same softmax logic as before.

Difference:

* (h_i) is no longer an average
* It is the result of a chain of transformations

---

## 8. What training means here (conceptually)

Training adjusts:

* (W_x): how input affects state
* (W_h): how memory persists
* (E): how tokens inject information

Errors at time (i) affect:

* current state
* previous states
* earlier embeddings

This is called **backpropagation through time**, but mechanically it is just:

* chain rule applied repeatedly

---

## 9. Fundamental limitations of vanilla RNNs

Now the important part.

### 9.1 Vanishing gradients

Look again at:

[
W_h^{i-k}
]

If eigenvalues of (W_h) are:

* < 1 → gradients shrink exponentially
* > 1 → gradients explode

This means:

* Very old tokens stop influencing learning
* Long-range dependencies are hard

This is not a bug. It is linear algebra.

---

### 9.2 Fixed memory size

The state (h_i) has fixed dimension (d).

All history must be compressed into (d) numbers.

If information is lost, it is gone forever.

---

### 9.3 Sequential computation

You cannot compute:

* (h_5) before (h_4)

This makes RNNs:

* slow
* hard to parallelize

This matters at scale.

---

## 10. Why LSTM and GRU exist (brief, logical)

They modify:

[
h_i = g(h_{i-1}, x_i)
]

by adding:

* gates
* additive memory paths

Goal:

* prevent vanishing gradients
* preserve information longer

But they keep the **same recurrence idea**.

---

## 11. Why attention eventually replaced RNNs

RNN:

* information flows through time step by step
* memory decays

Attention:

* any token can directly access any other token
* no forced compression

But attention **only makes sense after you understand RNNs**.

---

## 12. Final stripped summary

* RNN introduces a persistent state
* State is updated sequentially
* Order is intrinsic
* Recency bias emerges naturally
* Training propagates errors backward through time
* Long-term memory is fragile
* Computation is sequential

That is the complete picture.
---