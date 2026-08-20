+++
date = '2026-08-18T21:14:00-07:00'
draft = false
title = 'How Far Can Appearance-Only Visual Memory Go? Evaluating SigLIP2 for Indoor Place Recognition'
tags = ["computer-vision", "place-recognition", "vision-language-models", "robotics"]
description = "A systematic evaluation of SigLIP2 as a visual memory backbone for indoor place recognition, probing how far appearance-only embeddings can take us before geometry becomes non-negotiable."
+++

To perform long-horizon tasks, agents need memory that supports reasoning over time and reduces the search space. In robotics there is substantial work on spatial and episodic memory, most of which combines geometry, pose, and tracking. This post asks a simpler question: **How far can a pure appearance-based approach using a modern vision-language embedding (SigLIP2) go for place recognition and basic memory under real object and viewpoint change?**

## Overview

I test a lightweight online appearance-based visual memory. Each image is embedded with [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-512) and either assigned to an existing place cluster or used to start a new one, based on cosine similarity. For a known place, previous observations can be retrieved. High-level change descriptions are generated with a vision-language model only as a secondary step.

**Explicit non-goals:** This system does not use camera pose, ego-motion, geometric verification, 3D structure, or persistent object identities. It is intentionally appearance-only.

## Dataset

I created a small controlled dataset of indoor object arrangements containing 26 images. It uses 6 recurring objects placed on 3 surfaces (two wooden tables and one marble surface). Each image contains at most 4 objects. Objects are deliberately added, removed, or rearranged, and images are captured from multiple viewpoints.

Images are labeled with a Place ID. From these images I constructed sequences that simulate a robot moving between places over time.

<div style="display:flex; gap:0.5rem; margin:1.5rem 0;">
<img src="/images/siglip2/000.jpg" alt="Scene A" style="width:33%; border-radius:6px;">
<img src="/images/siglip2/014.jpg" alt="Scene B" style="width:33%; border-radius:6px;">
<img src="/images/siglip2/020.jpg" alt="Scene C" style="width:33%; border-radius:6px;">
</div>

## Evaluation & Results

### Place Assignment

The first experiment tests whether SigLIP2 embeddings can retrieve images from the same place. For each image I query against the remaining images and measure Recall@K (the fraction of queries where at least one image from the same place appears in the top-K results).

On this dataset, **Recall@1 is 96% (25/26)** and **Recall@5 is 100%**.

The figure below shows top-5 retrievals for several queries. Green borders indicate the correct place; red borders indicate the wrong place. The final row highlights the single Recall@1 failure.

![Retrieval results](/images/siglip2/exp1_retrieval_viz.png)

In the failure case, a query from a wooden table retrieves an image from the marble surface with very high similarity (0.910). Both images contain nearly identical foreground objects (red scoop, red cube, and metal clip). The embedding matched the objects more strongly than the surface, showing that global features can become object-dominated.

The distribution of cosine similarities shows substantial overlap between same-place and different-place pairs. While same-place scores tend to be higher, different-place pairs can still reach 0.90+. This overlap explains why a simple threshold is insufficient and why object-dominated scenes can produce high-similarity false matches across places.

![Retrieval score distribution](/images/siglip2/exp1_score_distribution.png)

### Similarity Behavior Along a Sequence

I constructed a sequence that simulates a robot moving between the three places, with the goal of using SigLIP2 embeddings to retrieve a previous image from the same place.

The figure below shows the results. The top panel shows the ground-truth place sequence (each place has a distinct color). Panel (b) shows consecutive SigLIP2 similarity. These scores remain high across most steps, including many of the true place transitions (highlighted in red). The average consecutive similarity is 0.837. This indicates that a drop in consecutive similarity is not a reliable signal for detecting that the agent has moved to a new place, especially when different places share similar objects.

Panel (c) shows similarity to the first observation of each place. There is clear within-place drift: as objects are rearranged, similarity to the original anchor declines. When the sequence later returns to a previously seen place, similarity to the original anchor recovers to varying degrees — sometimes strongly, sometimes only moderately.

![Sequence similarity](/images/siglip2/exp2_task1_similarity.png)

Together, these results highlight two practical limitations of pure appearance-based memory:

1. Consecutive embedding similarity is a weak detector of place changes.
2. A single stored image per place becomes a progressively worse reference as the scene evolves.

## Change Detection

Once a previous observation is retrieved, generating a reliable description of what changed remains difficult. Free-form outputs from a vision-language model were frequently noisy, inconsistent, and prone to confusing viewpoint changes with object motion. This suggests that turning appearance-based memory into trustworthy change detection still requires stronger structure or additional geometric or instance-level information.

## Conclusion

A pure SigLIP2 appearance memory can achieve strong place assignment on a controlled indoor set, but fails in predictable object-dominated cases and provides weak transition signals. It can serve as a lightweight memory layer for high-level semantic context, but it is insufficient as a standalone solution for reliable robotic place recognition under change.

Clear next steps involve hybrid appearance + geometry approaches or instance-level association.