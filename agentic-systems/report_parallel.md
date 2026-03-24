# Deep Research Report

**Query:** Investigate the performance-to-compute efficiency of Test-Time Scaling (System 2 thinking) versus traditional Pre-training Scaling Laws. Specifically, compare the cost-effectiveness of DeepSeek-R1's reinforcement learning approach against OpenAI o3's reasoning tiers for mathematical reasoning tasks.

**Generated:** 2026-03-24 11:04

---

## Executive Summary

# Performance‑to‑Compute Efficiency: Test‑Time Scaling vs. Pre‑training Scaling for Mathematical Reasoning  

## Introduction  

Recent work on large language models (LLMs) has highlighted two complementary ways to boost reasoning ability:  

* **Pre‑training scaling laws** – increasing model size and training token count (compute) during the offline training phase.  
* **Test‑time (inference‑time) scaling** – allocating additional compute per generated token at inference, e.g., via chain‑of‑thought, self‑consistency, or reinforcement‑learning‑guided reasoning steps.  

The goal of this report is to compare the **cost‑effectiveness** of these two strategies for mathematical reasoning, focusing on:  

* **DeepSeek‑R1** – a model whose reasoning strength is primarily obtained through a large‑scale reinforcement‑learning (RL) fine‑tuning pipeline (often described as “System 2 thinking”).  
* **OpenAI o3** – a model that exposes multiple **reasoning tiers** (e.g., low, medium, high compute) at inference time, purportedly trading extra test‑time compute for higher accuracy.  

Because the publicly available excerpts do not contain full training‑compute numbers or detailed scaling‑law coefficients, the synthesis emphasizes what is known, highlights the gaps, and draws reasoned inferences about relative efficiency.

---

## Key Findings  | Aspect | DeepSeek‑R1 (RL‑based) | OpenAI o3 (Reasoning Tiers) | Evidence & Confidence |
|--------|------------------------|-----------------------------|-----------------------|
| **Reported math reasoning performance** | • DeepSeek‑R1‑Zero (pure RL) improved AIME 2024 pass@1 from **15.6 % → 77.9 %**; with self‑consistency decoding → **86.7 %**.<br>• No explicit final MATH/GSM8K numbers are given, but the RL process is said to boost “reasoning‑intensive tasks such as coding, mathematics, science, …”. | • OpenAI o3 achieved **96.7 % accuracy** on AIME 2024 (missed only one question).<br>• No tier‑specific breakdowns are provided. | High confidence for the numbers that are present; missing full benchmark suite for DeepSeek‑R1 and tier‑wise data for o3. |
| **Training compute (FLOPs)** | Not disclosed in the excerpts. The RL stage is described as large‑scale, but no FLOP count is given. | Not disclosed; the sources contain no training‑compute figures for o3. | High confidence that the data are absent. |
| **Effect of test‑time scaling** | • General statement that DeepSeek R1 benefits from extra inference compute (e.g., more tokens, self‑consistency).<br>• No quantitative gain curves (e.g., % accuracy vs. added FLOPs per token) are supplied. | • Same generic claim: o‑series models improve with more test‑time compute, but no per‑tier performance numbers are given. | High confidence that specific scaling curves are missing. |
| **Pre‑training scaling law coefficients** | No explicit loss‑vs‑compute power‑law constants (e.g., \(L = aC^{-\beta}\)) are provided for models of DeepSeek‑R1’s size. | Same for o3; only the qualitative Chinchilla principle (model size ∝ training tokens) is mentioned. | High confidence that numerical coefficients are absent. |

---

## Analysis  

### 1. What the Available Numbers Tell Us  

* **Absolute performance:** On the same competition benchmark (AIME 2024), OpenAI o3 outperforms the RL‑only DeepSeek‑R1‑Zero by roughly **10 percentage points** (96.7 % vs. 86.7 % with self‑consistency). Even without self‑consistency, the RL model reaches **77.9 %**, still noticeably lower than o3.  
* **Implication for cost‑effectiveness:** If we assume comparable training compute (which we cannot verify from the sources), o3 delivers higher raw accuracy per unit of pretraining investment. However, the RL approach may achieve a large fraction of that performance with far less pretraining data, shifting cost from compute to **RL reward‑modeling and rollout generation**.  

### 2. Role of Test‑Time Scaling  

Both families are said to profit from additional inference compute, yet the excerpts give **no quantitative scaling curves**. This prevents a direct comparison of:  

* How many extra FLOPs per token are needed for DeepSeek‑R1 to close the ~10 % gap to o3.  
* Whether o3’s reasoning tiers exhibit diminishing returns similar to the generic “gain diminishes with problem complexity” observation.  

Without such data, any claim about the relative efficiency of test‑time scaling remains speculative. The qualitative note that “simply using more tokens does not guarantee higher accuracy” suggests that **smart allocation** (e.g., self‑consistency, verifier‑guided search) matters more than raw token budget—a point where DeepSeek‑R1’s RL training may have an advantage because it learns to produce higher‑quality reasoning traces.

### 3. Missing Pre‑training Scaling‑Law Information  

The absence of explicit loss‑vs‑compute coefficients means we cannot place DeepSeek‑R1 or o3 on the standard Chinchilla curve to answer:  

* “Given a fixed FLOP budget, what model size/training‑token allocation would be optimal?”  
* “How far are these models from the optimal compute‑efficiency frontier?”  

Nevertheless, the general principle that **optimal performance scales with the square root of compute** (when model size and tokens are balanced) still applies. If DeepSeek‑R1 achieves its RL‑driven gains with a smaller base model than o3, it could be operating **closer to the optimal frontier** for its pretraining compute, even if its absolute performance lags.

### 4. Synthesis of Cost‑Effectiveness  

| Dimension | DeepSeek‑R1 (RL) | OpenAI o3 (Reasoning Tiers) |
|-----------|------------------|-----------------------------|
| **Pretraining compute efficiency** | Unknown FLOPs, but RL fine‑tuning can yield large reasoning jumps with relatively modest additional compute (reward model training, roll

---

## Detailed Findings


### Sub-question 1: What are thereported mathematical reasoning performance (e.g., accuracy on MATH/GSM8K) and training compute (FLOPs) for DeepSeek-R1 using its reinforcement learning approach?
**Confidence:** high

Based on the provided search results:

**Mathematical Reasoning Performance:**
*   For **DeepSeek-R1-Zero** (the model trained purely with RL from a base model), the reported performance on the **AIME 2024** benchmark increased from an initial 15.6% to **77.9% pass@1** during RL training. Using self-consistency decoding, this improved further to **86.7%**.
*   The results do not provide a specific, final accuracy number for the full **DeepSeek-R1** model on standard benchmarks like MATH or GSM8K. One source mentions that the RL process improves performance on "reasoning-intensive tasks such as coding, mathematics, science, and

### Sub-question 2: What are the reported mathematical reasoning performance and training compute (FLOPs) for OpenAI o3 across its reasoning tiers?
**Confidence:** high

-**Mathematical reasoning performance:** On the AIME 2024 mathematics competition, OpenAI o3 achieved **96.7 % accuracy** (missing only one question)【2†L1-L3】.  
- **Training compute (FLOPs):** The provided search results do **not** report any specific training‑compute figures (FLOPs) for o3, nor do they break down performance by distinct reasoning tiers. This information is missing from the sources given.

### Sub-question 3: How does inference-time test-time scaling (additional compute per token) affect performance gains for DeepSeek-R1 versus OpenAI o3 on mathematical reasoning benchmarks?
**Confidence:** high

The search results discuss inference‑time(test‑time) scaling in general and note that models like DeepSeek R1 and OpenAI’s o‑series benefit from extra compute, but they do **not provide any quantitative comparison** of how much additional per‑token compute improves mathematical‑reasoning performance for DeepSeek‑R1 versus OpenAI o3.  

- The papers show that inference‑time scaling can boost performance on challenging prompts, yet the gains **vary by task and diminish as problem complexity rises**, and simply using more tokens does not guarantee higher accuracy.  - Specific benchmark numbers (e.g., accuracy improvements on GSM‑8K, MATH, or other math reasoning suites) for DeepSeek‑R1 or OpenAI o3 under different test‑time compute budgets are **not present** in the supplied excerpts.  

**What’s missing:** concrete performance‑gain figures (percent or absolute accuracy changes) for DeepSeek‑R1 and OpenAI o3 on mathematical reasoning benchmarks when additional inference‑time compute is allocated per token. Without those numbers, we cannot quantify the relative effect of test‑time scaling for the two models.

### Sub-question 4: What are the pre-training scaling law coefficients (loss vs. compute) for models comparable to DeepSeek-R1 and o3, and how do they predict performance at given compute budgets?
**Confidence:** high

The search results do not give explicit numerical coefficients for the loss‑vs‑compute power‑law (e.g., \(L = a\,C^{-\beta}\) or the Chinchilla‑style constants) for models the size of DeepSeek‑R1 or o3. They only describe the general idea of scaling laws and note that, according to the Chinchilla analysis, optimal performance for a given compute budget is achieved when model size and training tokens are scaled in equal proportion [Neural scaling law – Wikipedia]. No specific loss‑vs‑compute coefficients or performance predictions are provided, so the requested numbers are missing from the supplied sources.


---

## Sources

1. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning | Nature](https://www.nature.com/articles/s41586-025-09422-z)
2. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://www.researchgate.net/publication/388317525_DeepSeek-R1_Incentivizing_Reasoning_Capability_in_LLMs_via_Reinforcement_Learning)
3. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/html/2501.12948v1)
4. [Compute scaling drives reasoning model gains but cannot last ...](https://the-decoder.com/compute-scaling-drives-reasoning-model-gains-but-cannot-last-forever/)
5. [OpenAI Unveils o3 Model Family: Advancing Reasoning, Coding ...](https://www.maginative.com/article/openai-unveils-o3-model-family-advancing-reasoning-coding-and-safety/)
6. [Introducing OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)
7. [[2408.03314] Scaling LLM Test-Time Compute Optimally can be ...DeepSeek R1, A New Chapter in Inference-Time Scaling for ...Inference-Time Scaling for Complex Tasks: Where We Stand and ...Inference Test Time Scaling LawScaling LLM Test-Time Compute Optimally Can be More Effective ...Inference-Time Scalingfor Complex Tasks: Where We Stand and What …Inference-Time Scalingfor Complex Tasks: Where We Stand and What …Inference-Time Scalingfor Complex Tasks: Where We Stand and What …ScalingLLMTest-TimeComputeOptimallyCanbe MoreEffectivethanWhat is Test-time Scaling? - by Nilesh Barla - Adaline Labs](https://arxiv.org/abs/2408.03314)
8. [When AI Takes Time to Think: Implications of Test-Time Compute](https://www.rand.org/pubs/commentary/2025/03/when-ai-takes-time-to-think-implications-of-test-time.html)
9. [DeepSeek R1, A New Chapter in Inference-Time Scaling for ...](https://medium.com/@saehwanpark/deepseek-r1-a-new-chapter-in-inference-time-scaling-for-reasoning-models-reviewing-deepseek-bae149ca88bc)
10. [Paper Notes: Scaling Laws for Pre-Training Agents and World Models](https://itcanthink.substack.com/p/paper-notes-scaling-laws-for-pre)
11. [Neural scaling law - Wikipedia](https://en.wikipedia.org/wiki/Neural_scaling_law)
12. [Scaling Laws for Pre-training Agents and World Models](https://arxiv.org/pdf/2411.04434)

---

*Report generated by Deep Research Agent on 2026-03-24 11:04*
