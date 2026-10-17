# Academic Presentation Deck: Reinforcement Learning in Character-Level Phonetic Transliteration

**Course**: Reinforcement Learning (COMP 472), B.Tech in Artificial Intelligence  
**Institution**: Kathmandu University (KU), Department of Artificial Intelligence, School of Engineering  
**Authors**: Sundeep Dawadi & Shiv Ram Saud  
**Supervisor**: Prof. Dr. Manoj Shakya  
**Repository**: `phonetic_rl_study`

---

## Slide 01: Title & Academic Context
### Reinforcement Learning in Character-Level Phonetic Transliteration
#### Formulating Romanized Nepali to Devanagari Script Conversion as an Autoregressive Markov Decision Process (MDP)

- **Authors**: Sundeep Dawadi & Shiv Ram Saud
- **Department**: Department of Artificial Intelligence, School of Engineering
- **Institution**: Kathmandu University (KU), Dhulikhel, Nepal
- **Supervisor**: Prof. Dr. Manoj Shakya
- **Academic Scope**: Evaluation of policy gradients, state-space models (Mamba), gated convolutions, and composite phonetic rewards to eliminate exposure bias and resolve orthographic polysemy.

---

## Slide 02: Problem Formulation
### Why Romanized Nepali Transliteration is Non-Trivial

1. **Informal Typing Conventions**: Millions of Nepali speakers write Devanagari using English ASCII keyboards in chat, SMS, and search queries (e.g., `nepal`, `namaskar`, `kitab`).
2. **Alphabet Asymmetry**:
   - English QWERTY keyboard has 26 unaccented Latin letters.
   - Devanagari script features 33 primary consonants, 12 vowel diacritics (*matras*), 11 independent vowels, and complex conjunct ligatures (*samyuktakshar*).
3. **Core Challenge**: Mapping a compact, ambiguous Latin string into grammatically valid Devanagari characters without native phonological loss.

---

## Slide 03: Linguistic Complexities in Roman-to-Devanagari
### Schwas, Halantas, Ligatures, & Pluricentric Variants

- **Implicit Schwa Deletion**: Standalone consonants carry an inherent vowel */a/* (e.g., `क` = /ka/). Spoken Nepali systematically drops word-final and word-medial schwas, leading users to type `ghar` for `घर` or `kitab` for `किताब`.
- **Halanta (Virama `्`) Disambiguation**: Determining whether an English consonant clusters into a ligature (`स` + `्` + `क` = `स्क` in `namaskar`) or carries an explicit vowel.
- **Acoustic Homophones**:
  - Sibilants: `स` (dental), `श` (palatal), `ष` (retroflex) are distinct orthographically but pronounced identically in standard modern Nepali.
  - Vowels: *Hraswa* (short `ि`, `ु`) vs. *Dirgha* (long `ी`, `ू`).
  - Labials: `व` vs. `ब`.

---

## Slide 04: The Markov Decision Process (MDP) Formulation
### Character-Level Autoregressive Generation

We formalize character-level sequence generation as an episodic Markov Decision Process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

- **State Space ($\mathcal{S}$)**:
  $$s_t = \left( X, y_{<t} \right) = \left( [x_1, \dots, x_M], [y_0, y_1, \dots, y_{t-1}] \right)$$
  The complete Roman source sequence $X$ paired with the generated Devanagari character prefix $y_{<t}$.
- **Action Space ($\mathcal{A}$)**:
  $$\mathcal{A} = \mathcal{V}_{\text{tgt}} \cup \{\langle\text{EOS}\rangle, \langle\text{PAD}\rangle\} \quad (|\mathcal{A}| = 74)$$
- **Transition Dynamics ($\mathcal{P}$)**:
  Deterministic string concatenation $s_{t+1} = (X, [y_{<t}, a_t])$.
- **Terminal Condition**: When $a_t = \langle\text{EOS}\rangle$ or $t = T_{\max} = 22$.

---

## Slide 05: The Limitations of Supervised MLE
### Exposure Bias & Metric Mismatch

1. **Exposure Bias**:
   - Supervised models are trained with **Teacher Forcing**: at every step $t$, the decoder receives the true gold previous character $y_{t-1}^*$.
   - At test time, ground truth is absent; the model conditions on its own prior predictions $\hat{y}_{t-1}$.
   - A single corrupted early character cascades throughout the remaining sequence.
2. **Metric Mismatch**:
   - Cross-entropy minimizes token log-perplexity $\mathcal{L}_{\text{MLE}} = -\sum_{t} \log P(y_t^* \mid y_{<t}^*, X)$.
   - Real-world quality is measured on complete strings by Character Error Rate (CER), Phonetic Error Rate (PER), and Exact Match (EM).
   - Cross-entropy penalizes plausible alternate spellings equally to nonsensical gibberish.

---

## Slide 06: Reinforcement Learning Post-Training
### Policy Gradient Formulations

To optimize sequence-level rewards without teacher forcing:

1. **Self-Critical Sequence Training (SCST)**:
   - Samples an exploratory trajectory $Y^s \sim \pi_\theta(\cdot \mid X)$ and compares its reward against the greedy argmax baseline $Y^g = \arg\max \pi_\theta(\cdot \mid X)$:
     $$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}\left[ \left( R(Y^s) - R(Y^g) \right) \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t^s \mid y_{<t}^s, X) \right]$$
   - Unbiased, low-variance baseline that requires no auxiliary critic network.
2. **Advantage Actor-Critic (A2C)**:
   - Uses a token-level critic head estimating state value $V_\phi(s_t)$, computing TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
3. **Proximal Policy Optimization (PPO)**:
   - Clips probability ratios $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ within $[1-\epsilon, 1+\epsilon]$ to guarantee monotonic policy improvement.

---

## Slide 07: Reward Design & Sutton & Barto's Reward Hypothesis
### Formulations Tested

We systematically ablate five distinct reward formulations:

1. **Sparse Exact Match**: $R_{\text{EM}} = \mathbf{1}(Y = Y^*)$.
2. **Dense CER**: $R_{\text{CER}} = \max(0, 1 - \text{CER}(Y, Y^*))$.
3. **Composite Balanced**:
   $$R_{\text{Comp}} = 0.35 R_{\text{CER}} + 0.30 R_{\text{PER}} + 0.20 R_{\text{EM}} + 0.15 R_{\text{Prefix}}$$
4. **Dominant Phonetic**:
   $$R_{\text{Phon}} = 0.70 R_{\text{PER}} + 0.20 R_{\text{CER}} + 0.10 R_{\text{EM}}$$
5. **Orthotactic Penalties**: Penalizes illegal consecutive halantas (`््`) or mid-word independent vowels.

---

## Slide 08: Neural Sequence Architectures
### Comparative Architectural Landscape

We benchmark four distinct model families under identical vocabulary and parameter constraints:

1. **Gated ConvSeq2Seq GLU (5.23M)**:
   - 6 Gated Convolutional Encoder layers + 6 Causal Decoder layers with Gated Linear Units (GLU).
   - Fast parallel convolutions with receptive fields covering character n-grams.
2. **IndicXlit Transformer (11.13M)**:
   - 6 Encoder + 6 Decoder Pre-LN Transformer layers, $d_{\text{model}}=256, d_{\text{ff}}=1024, 4\text{ heads}$. Matches AI4Bharat IndicXlit dimensions.
3. **Mamba Selective SSM (2.02M)**:
   - Selective State Space Model with input-dependent discretization matrices $(\mathbf{B}, \mathbf{C}, \Delta)$ and linear $O(L)$ recurrence. Extremely lightweight.
4. **BiLSTM + Bahdanau Attention (7.37M)**:
   - 3-layer bidirectional LSTM encoder with additive Bahdanau attention. Classical recurrent baseline.

---

## Slide 09: Phonetic Error Rate (PER) & Equivalence Classes
### Measuring Acoustic vs. Orthographic Fidelity

Standard Character Error Rate (CER) treats all character substitutions equally. We define **Phonetic Error Rate (PER)** by mapping Devanagari characters into acoustic equivalence classes before Levenshtein alignment:

- **Sibilants**: $\{\text{स}, \text{श}, \text{ष}\} \to \mathbf{S}$
- **Vowel Quantity**: $\{\text{ि}, \text{ी}\} \to \mathbf{I}$, $\{\text{ु}, \text{ू}\} \to \mathbf{U}$
- **Labials**: $\{\text{व}, \text{ब}\} \to \mathbf{V}$
- **Nasals**: $\{\text{न}, \text{ण}, \text{ङ}, \text{ञ}, \text{ँ}, \text{ं}\} \to \mathbf{N}$
- **Result**: PER isolates genuine phonetic failures from benign pluricentric orthographic variations.

---

## Slide 10: Training Curriculum & Experimental Setup
### Two-Phase Methodology

1. **Phase 1: Supervised Pretraining (MLE)**:
   - Pretrained on clean Roman-Devanagari parallel pairs (from HuggingFace `Saugatkafley/Nepali-Roman-Transliteration`).
   - AdamW optimizer, inverse square root learning rate schedule, cross-entropy loss.
2. **Phase 2: Reinforcement Learning Fine-Tuning**:
   - Initialized from pretrained checkpoints.
   - Temperature sampling $\tau = 1.0$, full autoregressive rollouts without teacher forcing.
   - Gradients computed via SCST, A2C, and PPO on sequence rewards.

---

## Slide 11: Master Empirical Benchmark Results
### Quantitative Evaluation on 2,804 Held-Out Blind Test Samples

| Model & Checkpoint | Architecture | Parameters | Greedy Top-1 | Beam Top-1 | Beam Top-5 (↑) | CER (↓) | PER (↓) | MRR (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gated ConvSeq2Seq GLU (Clean)** | Gated CNN | 5.23M | **61.98%** | **62.45%** | **91.19%** | **0.0714** | **0.0494** | **0.7432** |
| **Transformer Clean (IndicXlit)** | Transformer | 11.13M | 57.99% | 58.27% | **89.94%** | 0.0789 | 0.0566 | 0.7119 |
| **Mamba SFT Continual** | Selective SSM | 2.02M | 52.00% | 52.78% | **85.84%** | 0.0965 | 0.0682 | 0.6586 |
| **Mamba Clean Pretrained** | Selective SSM | 2.02M | 51.89% | 52.89% | **85.63%** | 0.0958 | 0.0675 | 0.6598 |
| **Mamba RL (PPO Policy)** | Selective SSM | 2.02M | 51.43% | 52.35% | **85.59%** | 0.0959 | 0.0672 | 0.6572 |
| **Mamba RL (SCST Policy)** | Selective SSM | 2.02M | 48.32% | 49.57% | **83.49%** | 0.1021 | 0.0727 | 0.6328 |
| **Mamba RL (Actor-Critic A2C)** | Selective SSM | 2.02M | 49.36% | 49.96% | **82.77%** | 0.1025 | 0.0718 | 0.6293 |
| **BiLSTM + Bahdanau Baseline** | Recurrent | 7.37M | 37.59% | 39.37% | **72.90%** | 0.1311 | 0.0922 | 0.5256 |
| **Transformer Augmented (Noisy MLE)** | Transformer | 11.13M | 15.19% | 23.22% | **50.50%** | 0.2666 | 0.1701 | 0.3314 |

---

## Slide 12: Linguistic Orthographic Polysemy
### Case Study on Roman Input `rama`

In Roman typing, character mappings are inherently **one-to-many**. For example, `rama` validly maps to four distinct Devanagari words:

1. **राम** (*Ram*, masculine proper noun / deity) — in spoken Nepali, final schwas are elided; users type `rama` or `ram`.
2. **रामा** (*Rama*, feminine / classical vocative) — epic poetry (*"हे रामा"*), clan nomenclature.
3. **रमा** (*Rama*, past stem of verb *रमाउनु*) — meaning "delighted / rejoiced" (*"उनी बगैँचामा रमा"*).
4. **रम** (*Rum*, loanword) — colloquial beverage loanword or alternative schwa stem.

#### The Evaluation Dilemma
- If the test set annotator chose `राम`, a model outputting `रामा` receives **0% Top-1 Exact Match**.
- Evaluating **Top-5 Beam Accuracy** surges accuracy from **52%–62% up to 85.6%–91.2%**, capturing all valid linguistic variants within a single tap!

---

## Slide 13: Empirical Noise Filtering Effect
### How Reinforcement Learning Filters Out Synthetic Training Noise

- **Supervised MLE Failure**: When trained on noisy augmented data, MLE uniformly memorizes spelling corruptions, collapsing to **11.77% accuracy** and **0.3359 CER** on the clean test set.
- **SCST RL Recovery**: Fine-tuning the exact same model with SCST policy gradients surges accuracy to **27.92% (+16.15% absolute gain)** and slashes CER to **0.1611 (-52.0% relative reduction)**.
- **Why**: Whole-word rollouts condition on the reward signal. Degraded trajectories receive negative advantage and are penalized, guiding policy mass back toward clean, standard Devanagari ligatures.

---

## Slide 14: Qualitative Prediction Analysis
### Real-Word Case Studies

| Roman Input | Gold Reference | Augmented MLE | SCST RL (Augmented) | Clean Supervised | Qualitative Linguistic Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sarwasadharansammale` | **सर्वसाधारणसम्मले** | शर्वशढरणसमले | **सर्वसाधारणसम्मले** (✓) | **सर्वसाधारणसम्मले** (✓) | RL corrected sibilant corruption and restored geminate halanta `म्म`. |
| `feraune` | **फेराउने** | फेरुणे | **फेराउने** (✓) | **फेराउने** (✓) | RL eliminated retroflex nasal and recovered causative verb root. |
| `prakritianusarko` | **प्रकृतिअनुसारको** | परकऋटिँुशरको | **प्रकृतिअनुसारको** (✓) | **प्रकृतिअनुसारको** (✓) | RL reconstructed complex conjuncts (`प्र`, `कृ`) from collapsed ligatures. |
| `paristhitiko` | **परिस्थितीको** | पऋशठीतीको | **परिस्थितिको** | **परिस्थितिको** | Gold has historical *dirgha*; RL output modern standard *hraswa* (LevSim 0.92). |

---

## Slide 15: Input Method Editor (IME) Paradigm
### Real-World Virtual Keyboard Usability

In commercial virtual keyboards (Google Indic Keyboard, Hamro Patro, SwiftKey):
- Transliteration models never output a solitary string; they present a **5-candidate recommendation strip**.
- For `rama`, our beam decoder produces:
  $$\text{Ribbon}(\texttt{"rama"}) = \left[ \text{राम (52\%)}, \text{ रामा (24\%)}, \text{ रमा (14\%)}, \text{ रम (7\%)}, \text{ राम्र (3\%)} \right]$$
- Across all models, **Top-5 Beam Search provides a +28.7% to +33.5% usability leap**, rendering the system practical for daily typing.

---

## Slide 16: Hardware Efficiency & Latency Profiling
### Performance on Apple Silicon Unified Memory

- **Mamba Selective SSM (2.02M)**:
  - Fastest parameter footprint (2.02M vs 11.13M).
  - Linear $O(L)$ recurrence: under 3.5 ms per inference step.
  - Process RSS RAM footprint: **~210 MB**.
- **Gated ConvSeq2Seq GLU (5.23M)**:
  - 23.5 ms per sample on Apple Silicon MPS (33% faster than 11.13M Transformer).
  - Best accuracy-to-compute ratio in the study.
- **Length Capping ($T_{\max} = 22$)**:
  - Matched the 99th percentile of Nepali word lengths, eliminating memory fragmentation and preventing degenerate empty rollouts.

---

## Slide 17: Academic Conclusion & Future Work
### Summary of Contributions

1. **MDP Formulation**: Successfully established character-level transliteration as an autoregressive Markov Decision Process in pure PyTorch across four neural architectures.
2. **Linguistic Polysemy Resolution**: Demonstrated that Top-5 Beam search ($k=5$) resolves inherent script ambiguities (e.g., `rama`), lifting usability from **52%–62% to 85.6%–91.2%**.
3. **Phonetic Error Rate (PER)**: Proved that >42% of character mismatches are harmless acoustic homophones, reducible via phonetic equivalence normalization.
4. **RL Noise Pruning**: Verified that sequence-level policy gradient rollouts (SCST) recover **+16.15% accuracy** on corrupted data by penalizing degraded spelling trajectories.

#### Future Work
- Contextual sentence-level decoding with neural language model rescorers.
- Direct Preference Optimization (DPO) on human keyboard typing logs.

---

### Attribution
- **Students**: Sundeep Dawadi & Shiv Ram Saud  
- **Course**: Reinforcement Learning (COMP 472), B.Tech in Artificial Intelligence  
- **Department**: Department of Artificial Intelligence, School of Engineering, Kathmandu University  
- **Supervisor**: Prof. Dr. Manoj Shakya  
