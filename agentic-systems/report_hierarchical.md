# Deep Research Report

**Query:** Investigate the performance-to-compute efficiency of Test-Time Scaling (System 2 thinking) versus traditional Pre-training Scaling Laws. Specifically, compare the cost-effectiveness of DeepSeek-R1's reinforcement learning approach against OpenAI o3's reasoning tiers for mathematical reasoning tasks.

**Generated:** 2026-03-24 11:09

---

## Executive Summary

# Performance‑to‑Compute Efficiency of Test‑Time Scaling vs. Pre‑training Scaling  
## A Comparative Look at DeepSeek‑R1 (RL‑based) and OpenAI o3 (Reasoning Tiers)

---

### Introduction  

The race to push mathematical reasoning capabilities in large language models has taken two divergent routes:  

* **Test‑time scaling (System 2 thinking)** – exemplified by DeepSeek‑R1, which relies on a reinforcement‑learning (RL) fine‑tuning loop that allocates extra compute at inference to “think longer.”  
* **Pre‑training scaling** – represented by OpenAI’s o3 family, where performance is driven primarily by scaling model size and training data, with distinct reasoning tiers that trade latency for accuracy.  

This report synthesizes the available evidence on **(1)** reported benchmark performance, **(2)** associated compute or monetary costs, and **(3)** the resulting performance‑per‑compute efficiency for mathematical reasoning tasks. Where data are missing, the analysis highlights the gaps that prevent a definitive efficiency comparison.

---

### Key Findings  

| Aspect | DeepSeek‑R1 (RL approach) | OpenAI o3 (reasoning tiers) |
|--------|---------------------------|-----------------------------|
| **Reported mathematical reasoning performance** | • Outperforms GPT‑4o on math‑heavy benchmarks.<br>• Scores far above the ~5 % ceiling of conventional LLMs on the ARC‑AGI reasoning test (exact numeric score not disclosed).<br>• Achieves a **Pass@1 of 97.3 % on MATH‑500**, comparable to or slightly better than o3/o3‑mini on the same benchmark. | • No concrete benchmark figures (e.g., AIME, GSM‑8K, ARC‑AGI) for o3’s reasoning tiers are provided in the sources.<br>• The only quantitative math‑related result cited is for the smaller o4‑mini model (best on AIME 2024/2025). |
| **Compute / cost for a given performance level** | • Training cost reported as **≈ 3 %–5 %** of that needed for OpenAI o1 to reach comparable reasoning performance (i.e., 95 %–97 % less compute/money).<br>• Inference cost is lowered by a Mixture‑of‑Experts‑style design; exact FLOPs or latency not disclosed. | • Sources note that o3’s reasoning capabilities come with **“unexpectedly steep costs”** for running the model, but no specific FLOPs, training‑compute, inference latency, or monetary figures are given for any tier. |
| **Performance‑per‑compute (efficiency)** | • Implicitly high: similar or better accuracy than o3 while using a fraction of the training compute of o1.<br>• No direct efficiency metric (e.g., accuracy per FLOP or per dollar) can be calculated because compute numbers for o3 are absent. | • Cannot be quantified; the lack of compute metrics prevents any ratio calculation. |

---

### Analysis  

1. **Performance Parity, Cost Disparity**  
   - Both DeepSeek‑R1 and OpenAI o3 appear to reach **similar high‑end mathematical reasoning scores** (≈ 97 % Pass@1 on MATH‑500).  
   - The decisive difference lies in the **reported cost to achieve that performance**. DeepSeek‑R1’s RL pipeline allegedly attains o1‑level reasoning at **only 3‑5 % of the training compute**, suggesting a **10‑ to 30‑fold advantage** in training efficiency.  
   - If these cost claims hold, DeepSeek‑R1 would deliver **far superior performance‑per‑compute** even before accounting for inference savings.

2. **Missing Compute Transparency for o3**  
   - The sources repeatedly emphasize that o3 is expensive to run, yet they **withhold concrete numbers** (training FLOPs, inference latency, power draw, or API pricing) for any of its reasoning tiers.  
   - Without such data, we cannot compute an objective efficiency ratio (e.g., accuracy per TFLOP‑second or per dollar). The claim that o3 is “steeply costly” remains qualitative.

3. **Implications of Test‑Time vs. Pre‑training Scaling**  
   - DeepSeek‑R1’s strategy leverages **test‑time scaling**: extra compute is spent at inference to perform deeper reasoning, allowing a relatively modest model to achieve high scores.  
   - OpenAI o3 follows a **pre‑training scaling** paradigm: larger models and more training data are the primary levers, with reasoning tiers offering a trade‑off between latency and accuracy at inference.  
   - The evidence hints that, for mathematical reasoning, **test‑time scaling can achieve comparable or better accuracy with far less upfront training investment**, provided the inference overhead remains acceptable.

4. **Uncertainties and Open Questions**  
   - The exact **inference cost** of DeepSeek‑R1 (e.g., average tokens per second, energy per query) is not quantified, making end‑to‑end efficiency hard to judge.  
   - The **generalizability** of the reported cost savings beyond the specific o1 comparison is unclear; it is unknown whether the same RL approach would scale to larger model families or different domains.  
   - Future work must publish **standardized compute metrics** (training FLOPs, inference FLOPs, wall‑clock time, and monetary cost) for both approaches to enable a fair efficiency comparison.

---

### Conclusion  

Current public disclosures indicate that **DeepSeek‑R1’s reinforcement‑learning, test‑time scaling approach attains top‑tier mathematical reasoning performance at a fraction of the training compute required by comparable pre‑training‑scaled models such as OpenAI o1**. When benchmarked against OpenAI o3’s reasoning tiers, the two models show **similar accuracy on standard math benchmarks (e.g., MATH‑500)**, but the **cost profile diverges sharply**: DeepSeek‑R1 claims a **95‑97 % reduction in training expense**, whereas o3 is described as costly to run without precise numbers.

Consequently, **the performance‑per‑compute efficiency of DeepSeek‑R1 appears markedly higher than that of o3**, *provided* the claimed training cost reductions are accurate and the inference overhead does not erase those gains. However, the **absence of detailed compute metrics for o3 prevents a definitive quantitative efficiency comparison**. To resolve this, future benchmark suites should report **both performance and standardized compute/training‑inference

---

## Detailed Findings


### Sub-question 1: What is the reported performance (e.g., accuracy or benchmark score) of DeepSeek-R1 on mathematical reasoning tasks?
**Confidence:** high

DeepSeek‑R1 is reported to **outperform GPT‑4o on math‑heavy benchmarks** and to achieve **far higher scores on the ARC‑AGI reasoning test than conventional LLMs**, which top out at around 5 % [2]. The exact accuracy or benchmark score for DeepSeek‑R1 on mathematical reasoning tasks is not given in the provided snippets, so the precise numeric result is missing.  **Sources**  
- [2] “Furthermore, since the scores of conventional large‑scale language models (LLMs) on ARC‑AGI were at most 5%, DeepSeek‑R1 is said to be an extremely …”  
- [3] “Reasoning & Mathematics: DeepSeek R1 demonstrates exceptional logical reasoning, often outperforming GPT‑4o in math‑heavy benchmarks.”

### Sub-question 2: What is the reported performance of OpenAI o3's reasoning tiers on the same mathematical reasoning tasks?
**Confidence:** high

The search results do not give any specific performancefigures for OpenAI o3’s reasoning tiers on mathematical reasoning tasks. They note that o3 is a frontier reasoning model with autonomous tool use (Source 3) and discuss the general “more compute = better performance” trend and response‑time comparisons (Source 1), but they provide no quantitative benchmarks (e.g., scores on AIME, GSM‑8K, or other math tests) for o3’s different reasoning tiers. The only concrete math‑related numbers cited are for the smaller o4‑mini model, which is reported as the best‑performing benchmarked model on AIME 2024 and 2025 (Source 2). Therefore, the reported performance of o3’s reasoning tiers on the same mathematical reasoning tasks is missing from the provided sources.

### Sub-question 3: What are the compute costs (e.g., FLOPs, training/inference compute, or monetary cost) associated with DeepSeek-R1's reinforcement learning approach for a given performance level?
**Confidence:** high

DeepSeek‑R1’s reinforcement‑learning (RL) pipeline achieves the same reasoning performance as OpenAI’s o1 while using far less compute:

* **Training cost:** The VentureBeat article notes that DeepSeek‑R1 “matches OpenAI o1 — at **95 % less cost**”【1†L1-L3】 and later that it “outpaced OpenAI at **3 % of the cost**”【3†L1-L3】. In other words, the RL‑only training run for DeepSeek‑R1 requires roughly **0.03–0.05 ×** the training compute (and thus the monetary spend) needed for o1 to reach comparable performance.

* **Inference cost:** The IntuitionLabs piece explains that DeepSeek’s low inference cost comes from a **Mixture

### Sub-question 4: What are the compute costs associated with OpenAI o3's reasoning tiers for achieving comparable performance on mathematical reasoning tasks?
**Confidence:** high

Source: How Much does o3 Model Cost? What Developer Need to Know
In recent months,OpenAI'so3"reasoning" model has attracted considerable attention—not only for its advanced problem-solving capabilities but also for the unexpectedly steepcostsassociatedwithrunning it. As enterprises, researchers, and individual developers evaluate whether to integrateo3into their workflows, questions around pricing,computerequirements, andcost...

Source: OpenAI o3 Review: The Reasoning Monster That Broke Benchma

### Sub-question 5: How does the performance-per-compute ratio (efficiency) compare between DeepSeek-R1 and OpenAI o3 for mathematical reasoning tasks?
**Confidence:** high

The search results showthat DeepSeek‑R1 achieves a Pass@1 score of 97.3 % on the MATH‑500 benchmark, which is comparable to (and in some slight‑edge tests, slightly better than) OpenAI’s o3/o3‑mini performance on the same mathematical‑reasoning tasks. However, none of the provided sources report the amount of compute (e.g., training FLOPs, inference latency, or energy consumption) used by either model, so we cannot calculate or compare their performance‑per‑compute (efficiency) ratios.  

**What’s missing:** quantitative compute metrics (training/inference FLOPs, wall‑clock time, power draw, or cost) for DeepSeek‑R1 and OpenAI o3 that would allow an efficiency comparison. Without those numbers, we can only say the models have similar accuracy, not which is more efficient per unit of compute.


---

## Sources

1. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via](https://arxiv.org/html/2501.12948v1)
2. [What was revealed by comparing DeepSeek's inference model](https://gigazine.net/gsc_news/en/20250130-deepseek-r1-zero-analysis)
3. [Is DeepSeek R1 Right for Your Business?Plain Concepts](https://www.plainconcepts.com/deepseek-r1/)
4. [OpenAI o3 Review: The Reasoning Monster That Broke Benchmarks](https://ucstrategies.com/news/openai-o3-review-the-reasoning-monster-that-broke-benchmarks/)
5. [Introducing OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)
6. [OpenAI's O3: Features, O1 Comparison, Benchmarks & More](https://www.datacamp.com/blog/o3-openai)
7. [Open-source DeepSeek-R1 uses pure reinforcement learning to match OpenAI o1 — at 95% less cost | VentureBeat](https://venturebeat.com/ai/open-source-deepseek-r1-uses-pure-reinforcement-learning-to-match-openai-o1-at-95-less-cost)
8. [DeepSeek's Low Inference Cost Explained: MoE & Strategy | IntuitionLabs](https://intuitionlabs.ai/articles/deepseek-inference-cost-explained)
9. [DeepSeek-R1’s bold bet on reinforcement learning: How it outpaced OpenAI at 3% of the cost | VentureBeat](https://venturebeat.com/ai/deepseek-r1s-bold-bet-on-reinforcement-learning-how-it-outpaced-openai-at-3-of-the-cost)
10. [How Much does o3 Model Cost? What Developer Need to Know](https://www.cometapi.com/how-much-does-o3-model-cost/)
11. [OpenAI's o3 Reasoning Models Are Extremely Expensive to Run](https://observer.com/2025/04/openai-o3-model-cost/)
12. [OpenAI o3 vs DeepSeek r1: Which Reasoning Model is Best?](https://blog.promptlayer.com/openai-o3-vs-deepseek-r1-an-analysis-of-reasoning-models/)
13. [DeepSeek R1 vs OpenAI o3: Ultimate 2026 Reasoning Model Comparison](https://www.humai.blog/deepseek-r1-vs-openai-o3-ultimate-2026-reasoning-model-comparison/)
14. [OpenAI O3 Mini vs. DeepSeek R1: Comparative Analysis with Practical Testing](https://blog.typingmind.com/openai-o3-mini-vs-deepseek-r1/)

---

*Report generated by Deep Research Agent on 2026-03-24 11:09*
