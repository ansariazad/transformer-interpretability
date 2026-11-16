# 🔭 Transformer Interpretability — Mechanistic Analysis from Scratch in PyTorch

> **What is your model actually doing? Find out with 6 interpretability techniques — all from scratch.**

A complete implementation of 6 transformer mechanistic interpretability techniques: attention visualization, head classification, activation patching (causal tracing), logit lens, linear probing, and gradient saliency. Based on Anthropic's circuits research and the ROME paper.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Anthropic](https://img.shields.io/badge/Based_on-Anthropic_Circuits-6366F1)](https://transformer-circuits.pub)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ansariazad/transformer-interpretability?style=social)](https://github.com/ansariazad/transformer-interpretability/stargazers)

---

## 📺 Demo

```
$ python3 10_transformer_interpretability.py

[2] ATTENTION PATTERN VISUALIZATION
     12  34  56  78
12    █   ░   ░   ░
34    ▒   █   ░   ░
56    ░   ▒   █   ░
78    ░   ░   ▒   █    ← diagonal = previous-token head

[3] ATTENTION HEAD TYPE CLASSIFICATION
    Layer 0 Head 0: previous-token
    Layer 0 Head 1: diagonal (self)
    Layer 1 Head 0: induction-like
    Layer 2 Head 1: uniform

[4] LOGIT LENS
    Layer 0 : '23' (8.2%)    ← early layers: noisy
    Layer 1 : '47' (12.1%)
    Layer 2 : '12' (31.4%)
    Final   : '12' (31.4%)   ← converges to final answer

[5] LINEAR PROBE (even/odd concept)
    Probe accuracy: 100.0%   ← concept is linearly encoded!

[6] GRADIENT SALIENCY
    Token ids:  12  34  56  78  90  11  22  33
    Saliency :  ▁▂▄▇▇▅▃█                      ← token 33 most salient
```

---

## 📑 Table of Contents

- [Why Interpretability](#-why-interpretability)
- [The 6 Techniques](#-the-6-techniques)
- [Quick Start](#-quick-start)
- [Technique Deep Dives](#-technique-deep-dives)
- [Key Findings from Circuits Research](#-key-findings-from-circuits-research)
- [Roadmap](#-roadmap)
- [References](#-references)

---

## 💡 Why Interpretability?

| Without Interpretability | With Interpretability |
|-------------------------|-----------------------|
| "The model predicts X" | "Layer 2 Head 3 (induction head) causes prediction X" |
| Debug by trial-and-error | Activation patching locates the exact component |
| Unknown feature reliance | Gradient saliency shows which tokens matter |
| Black-box fine-tuning | Linear probes confirm concept encoding |
| Mystery failures | Logit lens shows where the model "changes its mind" |

---

## 🛠️ The 6 Techniques

| Technique | What It Reveals | Key Paper |
|-----------|----------------|-----------|
| **Attention Visualization** | Which tokens attend to which | Bahdanau et al., 2015 |
| **Head Classification** | Induction, prev-token, diagonal head types | Olsson et al., 2022 |
| **Activation Patching** | Which layer/head causally drives a prediction | Meng et al., 2022 (ROME) |
| **Logit Lens** | How prediction evolves through layers | nostalgebraist, 2020 |
| **Linear Probing** | Whether a concept is linearly encoded | Alain & Bengio, 2017 |
| **Gradient Saliency** | Input token importance for a prediction | Simonyan et al., 2013 |

---

## 🚀 Quick Start

```bash
git clone https://github.com/ansariazad/transformer-interpretability.git
cd transformer-interpretability

# Concepts mode (zero deps)
python3 10_transformer_interpretability.py

# Full mode
pip install torch
python3 10_transformer_interpretability.py
```

---

## 🔬 Technique Deep Dives

### Activation Patching (Causal Tracing)
```python
# From the ROME paper: which layer is causally responsible for a fact?
effects = activation_patch(
    model,
    clean_input=correct_prompt,      # "The Eiffel Tower is in Paris"
    corrupted_input=corrupted_prompt, # "The Eiffel Tower is in [MASK]"
    target_token=vocab["Paris"],
)
# effects: {"layer_0": +0.02, "layer_1": +0.41, "layer_2": +0.89}
# → Layer 2 is the causal bottleneck for this fact
```

### Logit Lens
```python
# Decode residual stream at each layer
logit_lens(model, tokens, vocab)
# Layer 0: 'cat' (8%)     ← random early
# Layer 1: 'dog' (15%)
# Layer 2: 'Paris' (71%)  ← model "knows" at layer 2
```

### Linear Probe
```python
probe = LinearProbe(d_model=64)
probe.train(hidden_states, labels=[0 if even else 1 for token in tokens])
accuracy = probe.accuracy(hidden_states, labels)
# 100% → concept is perfectly linearly encoded in hidden states
# 50%  → concept is not represented (or non-linearly)
```

---

## 🧠 Key Findings from Circuits Research

From Elhage et al. (2021) — the foundation of mechanistic interpretability:

> **"A transformer = sum of attention heads + MLP layers + bias terms, all writing to a shared residual stream."**

Key circuit types found in real transformers:
- **Induction heads** — copy-and-completion circuit (token[i] after token[j] → token after j's successor)
- **Previous-token heads** — attend to position t-1
- **Duplicate-token heads** — flag repeated tokens
- **Name Mover heads** — in GPT-2: move subject name to output position (ROME)

---

## 🗺️ Roadmap

- [ ] Sparse autoencoder features (Anthropic monosemanticity)
- [ ] Attention knockout (ablation experiments)
- [ ] Path patching (fine-grained causal tracing)
- [ ] Neuron visualization (max-activating examples)
- [ ] Interactive attention viewer (Streamlit)

---

## 📚 References

| Paper | Authors |
|-------|---------|
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | Elhage et al. (Anthropic), 2021 |
| [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | Olsson et al. (Anthropic), 2022 |
| [Locating and Editing Factual Associations (ROME)](https://arxiv.org/abs/2202.05262) | Meng et al., 2022 |
| [Zoom In: Circuits](https://distill.pub/2020/circuits/zoom-in/) | Olah et al. (Distill), 2020 |
| [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) | Bricken et al. (Anthropic), 2023 |

---

## ⭐ Support

Fascinated by what's inside transformers? **Star the repo** to help others find this!

[![GitHub stars](https://img.shields.io/github/stars/ansariazad/transformer-interpretability?style=for-the-badge)](https://github.com/ansariazad/transformer-interpretability/stargazers)

---

## 📄 License

MIT © 2026

---

## 👤 Author

**Azad Ansari** — Python Developer & AI Engineer, Mumbai

[![GitHub](https://img.shields.io/badge/GitHub-ansariazad-181717?logo=github)](https://github.com/ansariazad)
[![Portfolio](https://img.shields.io/badge/Portfolio-ansariazad.github.io-blue)](https://ansariazad.github.io)
