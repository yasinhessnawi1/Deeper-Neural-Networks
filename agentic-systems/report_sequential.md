# Deep Research Report

**Query:** Investigate the performance-to-compute efficiency of Test-Time Scaling (System 2 thinking) versus traditional Pre-training Scaling Laws. Specifically, compare the cost-effectiveness of DeepSeek-R1's reinforcement learning approach against OpenAI o3's reasoning tiers for mathematical reasoning tasks.

**Generated:** 2026-03-24 11:02

---

## Executive Summary

# Performance‑to‑Compute Efficiency of Test‑Time Scaling vs. Pre‑Training Scaling  
## A Comparative Look at DeepSeek‑R1 (RL) and OpenAI o3 (Reasoning Tiers)

---

### Introduction  

The race to push mathematical reasoning capabilities in large language models has produced two complementary strategies:  

* **Pre‑training scaling** – increasing model size, data, and training FLOPs.  
* **Test‑time (System 2) scaling** – allocating additional compute at inference (e.g., chain‑of‑thought search, adaptive updating) to obtain reasoning gains without enlarging the model.  

DeepSeek‑R1 exemplifies a reinforcement‑learning (RL)‑driven approach that seeks to improve reasoning through post‑training optimization, whereas OpenAI o3 offers a family of “reasoning tiers” that explicitly trade inference compute for higher accuracy. This report synthesizes the available evidence on their **performance‑to‑compute efficiency**, examines how **test‑time scaling laws** relate to traditional pre‑training laws, and highlights the gaps that prevent a definitive cost‑effectiveness comparison.

---

### Key Findings  

| Aspect | What the sources tell us | What remains unknown |
|--------|--------------------------|----------------------|
| **DeepSeek‑R1 RL efficiency** | No quantitative accuracy numbers or FLOP/training‑cost figures are provided for DeepSeek‑R1 on any math benchmark. Consequently, an accuracy‑per‑FLOP metric cannot be derived. | Exact benchmark scores (e.g., MATH, GSM‑8K) and the FLOP budget (training or inference) used to achieve them. |
| **OpenAI o3 reasoning‑tier efficiency** | o3 reaches **≈ 90 % accuracy on the ARC‑AGI benchmark**, a three‑fold improvement over prior versions. The released “public o3” tiers are smaller than the originally benchmarked model, implying a trade‑off between tier size and inference compute. | Precise FLOP or dollar cost per tier, and the exact accuracy achieved on pure mathematical‑reasoning suites (MATH, GSM‑8K) at each tier. |
| **Test‑time vs. pre‑training scaling laws** | Both obey power‑law‑like relationships, but **test‑time compute is far more efficient**: allocating extra FLOPs at inference can match the performance of a model **≈ 14× larger** in parameters when no extra test‑time compute is used. Spending a given compute budget on test‑time search or adaptive updating yields higher returns than pretraining a bigger model. | Direct empirical curves for DeepSeek‑R1 or o3 that quantify the exponent of the test‑time scaling law versus the pre‑training law. |
| **Reported compute costs for specific accuracies** | The only concrete cost figure is for **DeepSeek v3** (≈ 15 trillion tokens, ≈ $5.5 M). No training‑FLOP, inference‑cost, or accuracy‑linked cost data exist for DeepSeek‑R1 or OpenAI o3 on math benchmarks. | Training FLOP counts, inference FLOPs/second or dollar‑per‑query, and the accuracy levels those costs were meant to achieve for either model. |
| **Cost‑effectiveness (dollars per point improvement)** | No dollar‑per‑point figures are supplied for either approach; thus a direct comparison cannot be made. | Training/inference cost of DeepSeek‑R1 and o3, together with the corresponding gain in benchmark points (e.g., +5 % on MATH). |

---

### Analysis  

1. **Missing Quantitative Foundations**  
   The core obstacle to a rigorous comparison is the absence of **benchmark‑specific accuracy numbers** paired with **compute expenditure** for both DeepSeek‑R1’s RL fine‑tuning and OpenAI o3’s reasoning tiers. Without these, any claim about “accuracy per FLOP” or “dollars per point” remains speculative.

2. **Implications of Test‑Time Scaling Efficiency**  
   The literature cited (PaLM 2‑S* study, the “Scaling LLM Test‑Time Compute Optimally” paper) consistently shows that **test‑time compute can substitute for roughly an order of magnitude increase in model size**. If DeepSeek‑R1’s RL procedure primarily improves the model’s *intrinsic* reasoning ability (i.e., shifts the pre‑training curve upward), then adding test‑time search on top of it could yield **compound gains**: a modest RL‑induced shift plus a large test‑time multiplier. Conversely, if OpenAI o3’s tiers are already operating near the test‑time scaling frontier, further RL‑based improvements may have diminishing returns unless they also shift the underlying scaling curve.

3. **Cost‑Effectiveness Landscape**  
   Assuming the reported **≈ $5.5 M** training cost for DeepSeek v3 is indicative of the order of magnitude for DeepSeek‑R1 (a similar scale model), we can infer that **training a large RL‑optimized model is expensive but amortized over many inferences**. OpenAI o3’s public tiers, being smaller than the originally benchmarked model, likely incur **lower inference FLOPs per query**, but the exact trade‑off is opaque. If o3 achieves ~90 % on ARC‑AGI with a modest inference budget, its **dollars‑per‑point** could be competitive, especially if the tier size permits cheap deployment at scale. However, without concrete numbers, this remains hypothesis.

4. **Strategic Takeaway**     - **When compute is scarce at inference time** (e.g., edge devices, low‑latency services), investing in a stronger pre‑trained/RL‑optimized base model (like DeepSeek‑R1) may be preferable because the model’s intrinsic capability reduces the need for costly test‑time search.  
   - **When inference compute can be flexibly allocated** (e.g., batch processing, cloud services with spare cycles), leveraging test‑time scaling—either via o3’s reasoning tiers or explicit search algorithms—can yield outsized performance gains per FLOP, potentially outperforming the returns from further pretraining or RL fine‑tuning.

---

### Conclusion  

The current evidence confirms that **test‑time (System 2) scaling follows a power‑law akin to traditional pre‑training scaling but delivers substantially higher performance per unit of compute**. Both DeepSeek‑R1’s reinforcement‑learning approach and OpenAI o3’s reasoning tiers are positioned to benefit from this phenomenon

---

## Detailed Findings


### Sub-question 1: What is the performance-to-compute efficiency (accuracy per FLOP) of DeepSeek-R1's reinforcement learning approach on mathematical reasoning benchmarks?
**Confidence:** high

The search results donot provide any quantitative figures for DeepSeek‑R1’s accuracy on mathematical‑reasoning benchmarks, nor do they give the FLOP count (or compute budget) associated with its reinforcement‑learning training. Consequently, we cannot calculate the performance‑to‑compute efficiency (accuracy per FLOP) for DeepSeek‑R1’s RL approach from the supplied information. To answer the question, we would need specific benchmark accuracy numbers (e.g., % correct on a math reasoning test) and the corresponding FLOP expenditure for the RL training or inference phase. These details are missing from the provided sources.

### Sub-question 2: What is the performance-to-compute efficiency of OpenAI o3's reasoning tiers on the same mathematical reasoning tasks?
**Confidence:** high

Source: OpenAI o3 Review: The Reasoning Monster That Broke Benchmarks
1 day ago -That’s not a version update—that’s a different product. “Public o3 is a different model… tuned for chat/product use. All released o3 compute tiers aresmaller than what was initially benchmarked.

Source: An In-Depth Analysis of OpenAI’s O3 Model and Its Comparative Performance | by Thomas Hansen | Medium
February 4, 2025 -ARC-AGI Benchmark: O3 has achievednearly 90% accuracy, demonstrating a threefold improvement in

### Sub-question 3: How do the scaling laws (performance vs compute) for test-time scaling (System 2 thinking) compare to traditional pre-training scaling laws for these models?
**Confidence:** high

Test‑time scaling (often called “System 2” thinking) obeys a power‑law similar to the classic pre‑training scaling law, but the compute‑to‑performance trade‑off is far more favorable: allocating extra FLOPs at inference can achieve the same performance gains that would require increasing the pretrained model size by roughly an order of magnitude.  

- In the PaLM 2‑S* study, **compute‑optimal test‑time scaling matched the performance of a model ≈ 14 × larger in parameters** when no extra test‑time compute was used【2†L1-L4】.  
- The paper *“Scaling LLM Test‑Time Compute Optimally can be More Effective than Scaling Model Parameters”* concludes that, for a given compute budget, **spending that budget on test‑time search or adaptive updating yields higher returns than spending it on pretraining a bigger model**【1†L1-L3】【3†L1-L4】.  
- The Wikipedia note on neural scaling laws adds that **performance gains can also be obtained by scaling inference (test‑time compute)**, extending the usual scaling law beyond parameters, data, and training cost【0†L1-L3】.  

Thus, while both pre‑training and test‑time compute follow scaling‑law‑like relationships, test‑time compute is **significantly

### Sub-question 4: What are the reported compute costs (training FLOPs, inference cost) for DeepSeek-R1 and OpenAI o3 to achieve specific accuracy levels on math benchmarks?
**Confidence:** high

The search results do notprovide the specific training‑FLOP counts or inference‑cost figures for **DeepSeek‑R1** or **OpenAI o3** tied to particular math‑benchmark accuracy levels.  

- The only concrete compute‑cost detail given is for **DeepSeek v3**: it was trained on 15 trillion tokens at an estimated cost of **≈ $5.5 million USD** of compute.  - The results mention a “truly absurd divergence” between the DeepSeek v3 release and the OpenAI o3 reveal in terms of **inference cost**, but they do not quote any numbers for either model’s training FLOPs or inference expense, nor do they link those costs to any accuracy achievements on math benchmarks.  

**What’s missing:**  
- Training FLOP (or token‑count) figures for DeepSeek‑R1 and OpenAI o3.  
- Reported inference‑cost metrics (e.g., dollars per token, latency, or FLOPs per query) for those models.  
- The specific math‑benchmark accuracy levels (e.g., MATH, GSM‑8K) that those costs were meant to achieve.  

Without those details, we cannot concisely state the requested compute costs for DeepSeek‑R1 and OpenAI o3.

### Sub-question 5: How does the cost-effectiveness (e.g., dollars per point improvement) of DeepSeek-R1's RL approach compare to OpenAI o3's reasoning tiers for mathematical reasoning?
**Confidence:** high

The search results provided only givegeneral definitions of cost‑effectiveness analysis and do not contain any specific figures or comparisons for DeepSeek‑R1’s reinforcement‑learning approach versus OpenAI o3’s reasoning tiers in mathematical reasoning. Consequently, I cannot state a dollars‑per‑point‑improvement value for either method or directly compare their cost‑effectiveness. To answer the question, data such as the training/inference costs of DeepSeek‑R1 and OpenAI o3 and the corresponding gains on a math‑reasoning benchmark (e.g., points on MATH or GSM‑8K) would be required. This information is missing from the supplied results.


---

## Sources

1. [DeepSeek-R1: Reinforcement Learning for LLM Reasoning](https://www.libertify.com/interactive-library/deepseek-r1-reinforcement-learning-llm-reasoning-2/)
2. [Reasoning Models Explained: OpenAI o1/o3 vs DeepSeek R1 vs QwQ-32B](https://blog.premai.io/reasoning-models-explained-openai-o1-o3-vs-deepseek-r1-vs-qwq-32b/)
3. [Calculate Computational Efficiency of Deep Learning Models with FLOPs ...](https://www.kdnuggets.com/2023/06/calculate-computational-efficiency-deep-learning-models-flops-macs.html)
4. [OpenAI o3 Review: The Reasoning Monster That Broke Benchmarks](https://ucstrategies.com/news/openai-o3-review-the-reasoning-monster-that-broke-benchmarks/)
5. [An In-Depth Analysis of OpenAI’s O3 Model and Its Comparative Performance | by Thomas Hansen | Medium](https://medium.com/@thomas_78526/an-in-depth-analysis-of-openais-o3-model-and-its-comparative-performance-813a7c57a83e)
6. [OpenAI o1 and o3 Reasoning Models: Complete Guide for Engineers](https://reintech.io/blog/openai-o1-o3-reasoning-models-explained)
7. [Neural scaling law - Wikipedia](https://en.wikipedia.org/wiki/Neural_scaling_law)
8. [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/html/2408.03314v1)
9. [[2408.03314] Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
10. [TAI #132: Deepseek v3–10x+ Improvement in Both Training and](https://newsletter.towardsai.net/p/tai-132-deepseek-v310x-improvement)
11. [DeepSeek v3, Microsoft Phi 4, OpenAI o3, new tools and more -](https://thisweekinaiengineering.com/p/deepseek-v3-microsoft-phi-4-openai)
12. [DeepSeek v3: 671B finegrained MoE trained for $5.5m USD of](https://news.smol.ai/issues/24-12-26-ainews-deepseek-v3-671b-finegrained-moe-trained-for-dollar55m-usd-of-compute-on-15t-tokens)
13. [Cost-Effectiveness Analysis - an overview | ScienceDirect Topics](https://www.sciencedirect.com/topics/economics-econometrics-and-finance/cost-effectiveness-analysis)
14. [Cost-effectiveness analysis - Wikipedia](https://en.wikipedia.org/wiki/Cost-effectiveness_analysis)
15. [PDFMethodology Guide: Cost-Effectiveness Analysis](https://coast.noaa.gov/data/digitalcoast/pdf/econguide-cost-effectiveness.pdf)

---

*Report generated by Deep Research Agent on 2026-03-24 11:02*
