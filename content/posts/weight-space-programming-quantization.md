+++
date = '2026-11-11T01:35:00-08:00'
draft = false
title = 'Can an LLM Quantize a Model by Reading Its Weights? Testing Weight-Space Programming on Qwen3-4B'
tags = ["llm", "quantization", "interpretability", "weight-space-programming"]
description = "I tried using an LLM as a weight-space programmer to quantize Qwen3-4B to 2 and 3 bits. It found real structure in the weights, and that structure mattered. Then a one-line generic rule beat it."
+++

A recent post, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), argues that frontier LLMs can read and write neural network weights directly: tell what a network does from its raw numbers, fix it by editing a small fraction of them, even write small networks from scratch. Quantization looked like a natural test. It is a pure weight-space problem (decide how every number gets rounded, which ones to protect, where to spend bits), and there are strong baselines to compare against. This post asks: **if an LLM reads a model's weights, can it make better quantization decisions than generic algorithms?**

Short answer: it found real structure, protecting that structure helped, and at 3 bits the result beat GPTQ and AWQ on GSM8K. But when I ran the ablation, a generic one-line rule protecting the same number of weights did better still.

## Setup

- **Model:** Qwen3-4B, non-thinking mode. Only the linear layers inside the 36 decoder blocks are quantized; embeddings and norms stay in BF16.
- **Format:** asymmetric integer quantization in groups of 128 weights, each group with an FP16 scale and zero point. 2-bit costs 2.25 bits per weight (bpw), 3-bit costs 3.25. Any weight kept in BF16 is charged 16 bits.
- **Evals:** Wikitext-2 perplexity and GSM8K (zero-shot, greedy).
- **The weight-space programmer:** Claude Opus 5.5, working in a Python sandbox with the weights loaded. It could inspect weights with numpy and torch, but no quantization libraries, no gradients, no fine-tuning.

Two arms, mirroring the original post:

- **Arm A (weights only):** no data and no forward passes.
- **Arm B (weights + a budget):** at most 20 forward passes of 16 C4 sequences each. Every model call went through a counter that logged it and refused past the budget. Configurations were chosen on held-out C4 text, never on the test sets.

**Explicit non-goals:** this is not a new state of the art, and it is not an LLM editing weights one number at a time. The LLM reads the weights, reasons about them, and writes the quantization code.

## What the weights revealed

Reading the weights alone, before any data, turned up three things.

**A massive-activation channel.** Residual channel 0 receives about 2.4x the writes of a typical channel across all 72 output projections. Yet from layer 3 onward, the norm in front of every MLP scales channel 0 by roughly zero. The model stores something huge there and then hides it from the MLP. Channel 4 is a weaker version of the same thing.

![Massive channel](/images/wsp-quant/fig1_massive_channel.png)

**Super weights.** The largest weights in the model, around 45-55 standard deviations out, sit in the down-projection rows that write channels 0 and 4, in layers 1, 2, 6, 7, 10, 14 and 16.

**Norm scales that grow with depth,** from about 0.2 in layer 0 to 29 in layer 34. A weight's real effect is the weight times the scale in front of it, which plain rounding ignores.

One forward pass later, the activations confirmed it in a sharper form. In layers 6 and 16, **99% of the MLP's activation energy flows through 8 of its 9,728 neurons.**

![Super neurons](/images/wsp-quant/fig2_super_neurons.png)

## 2-bit: from collapse to broken

Plain round-to-nearest (RTN) at 2 bits destroys the model: Wikitext perplexity goes from 13.66 to 255,599, and the output is multilingual noise.

The weights-only arm brought that down to 528 at 2.26 bpw. Most of the gain came from a generic trick (searching each group for a tighter rounding range). The model-specific decisions (keep the massive-channel rows in BF16, weight rounding by the norm scale) gave another 7x on top.

Two lessons came out of this stage:

1. **Lower weight error can mean a worse model.** A least-squares refit of each group lowered the weight error and *raised* perplexity from 528 to 805. HQQ, a published calibration-free method, showed the same pattern at 3 bits: lower weight error than RTN, worse perplexity and GSM8K. At low bit widths, what matters is the error in each layer's output, and that depends on activations a weights-only method cannot see.
2. **Reading weights plus one diagnostic pass located the damage.** Comparing the 2-bit model against BF16 layer by layer, the error jumps at layers 1 and 16. I had protected the rows that *write* the massive channels, but not the super neurons that *produce* those values. Keeping 8 neurons per layer in BF16 cost 0.008 bpw and cut held-out perplexity by about 3x.

![Damage by layer](/images/wsp-quant/fig3_damage_by_layer.png)

With activation statistics, the best 2-bit model reached 65 perplexity at 2.38 bpw. That is still a model that starts reasoning sensibly and then loops on a phrase: 3% on GSM8K. Better rounding decisions alone could not get 2-bit to usable.

## 3-bit: it works

The same recipe at 3 bits gives a usable model. On the full GSM8K test set (1,319 problems):

| Method | bpw | Wikitext PPL | GSM8K |
|---|---|---|---|
| BF16 | 16 | 13.66 | 89.1% |
| RTN 4-bit | 4.25 | 16.10 | 85.3% |
| **LLM-guided (Arm B)** | 3.27 | 16.68 | **81.6%** |
| GPTQ, 128 calibration seqs | 3.25 | **15.11** | 76.7% |
| AWQ, 128 calibration seqs | 3.25 | 18.32 | 73.3% |

GPTQ and AWQ were run with their standard implementations in llm-compressor. Because every model answers the same problems, I compared them with a paired test (exact McNemar). The LLM-guided model beats GPTQ by 4.9 points and AWQ by 8.3 (both p < 0.0001). It stays below RTN 4-bit (p = 0.0007), which spends a full extra bit per weight.

Without any data, the weights-only arm also beat the data-free baselines on a 200-problem subset: 73.5% on GSM8K, against 57% for RTN and 49% for HQQ.

The most interesting pattern is that **perplexity and reasoning disagree.** GPTQ has the best perplexity of any 3-bit model and loses clearly on GSM8K.

![Perplexity vs GSM8K](/images/wsp-quant/fig4_ppl_vs_gsm8k.png)

## The ablation

This is the experiment that decides whether reading the weights mattered. I replaced each model-specific decision with a generic one at the same bit budget. The generic outlier rule keeps the 0.139% of weights with the largest |weight| x input RMS in BF16, exactly the number of weights Arm B keeps through massive channels and super neurons.

| 3-bit, same budget | Wikitext PPL | GSM8K |
|---|---|---|
| Generic only (activation-aware scaling and rounding range) | 17.75 | 80.4% |
| + massive channels and super neurons (Arm B) | 16.68 | 81.6% |
| + generic top-0.139% outlier rule instead | **16.44** | **84.7%** |

At 2 bits the gap is wider: the generic rule reaches 37.8 perplexity, against 85.6 for the model-specific version at the same 2.27 bpw.

So:

- The structure the LLM found is real, and protecting it helps compared with protecting nothing.
- On GSM8K, the model-specific decisions are not significantly better than generic-only (p = 0.23).
- A generic importance score protecting the same number of weights beats them at both bit widths. It presumably catches the same super neurons plus scattered outliers the hand-picked structure misses.
- The win over GPTQ comes from the generic activation-aware parts: generic-only already beats GPTQ (80.4% vs 76.7%, p = 0.002).

## Conclusion

Reading weights works as interpretability: from raw numbers, the LLM found the massive channel, the super weights and the layers where quantization breaks, and every one of those findings held up when checked against activations. As optimization, it did not beat a simple rule. Once the important weights are identified with a generic score that sees activations, hand-picked structure adds nothing. Weight-space reasoning also cannot see what matters at low bit widths, which is output error rather than weight error.

Two findings feel worth keeping regardless:

1. At 3 bits, activation-aware rounding plus a small set of protected weights beats GPTQ and AWQ on GSM8K, while losing to GPTQ on perplexity. Perplexity is not a safe proxy for reasoning when choosing a quantization method.
2. In Qwen3-4B, a handful of neurons in a few layers carry almost all of the MLP's activation energy. Any low-bit method has to protect them, whether it finds them by reasoning or by a score.

**Caveats:** one model, one run of the LLM in the loop, and a single GPTQ calibration seed (llm-compressor shuffles calibration samples; a re-run moved its perplexity from 15.38 to 15.11). The weights-only arm's settings were chosen on Wikitext during the 2-bit pilot, a small selection leak. The next steps are a second model family, and a control where the LLM gets only the architecture and not the weights, to separate reading the weights from remembering the quantization literature.
