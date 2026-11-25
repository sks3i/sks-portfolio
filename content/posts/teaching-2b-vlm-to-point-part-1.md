+++
date = '2026-11-24T21:00:00-08:00'
draft = false
title = 'Teaching a 2B VLM to Point, Part 1'
tags = ["vision-language-models", "reinforcement-learning", "grpo", "pointing", "robotics"]
description = "SFT and four GRPO runs on one RTX 4090 took Qwen3-VL-2B level with a 4-bit Qwen3-VL-32B on Point-Bench and RefCOCO, at a tenth of the latency. What the runs taught me about what RL can and cannot do."
+++

I spent a few weeks teaching a 2-billion-parameter vision-language model to point at things, on a single RTX 4090. It ended up statistically level with a 4-bit Qwen3-VL-32B on the two main pointing benchmarks, while answering about ten times faster.

The interesting part is not the final number. It is what four reinforcement learning runs taught me about what RL can and cannot do. The short version: **with a short RL budget like mine, RL sharpens behavior the model already sometimes produces; it did not create skills the model never showed.** That matches the main line of recent work ([Yue et al., 2025](https://arxiv.org/abs/2504.13837)), though much longer RL runs can push past it ([ProRL, 2025](https://arxiv.org/abs/2505.24864)). Almost everything below is that idea, learned the hard way.

## The goal

A point is the smallest useful answer a model can give about where something is. A robot needs a spot to grasp or a place to put a cup down. A UI agent needs a pixel to click. A point is also trivially checkable: it either lands on the object or it does not. That makes pointing a near-ideal playground for RL with a verifiable reward.

The model answers in one fixed format, `<point x="512" y="340">`, with coordinates scaled to 0–1000 so they do not depend on image size. I started from Qwen3-VL-2B-Instruct and set myself two constraints. Everything trains and evaluates on one 24 GB consumer GPU. And the model stays at 2B, because the end goal is on-device use, so any gain has to come from data, rewards and training rather than scale.

![One example per benchmark](/images/pointing-part1/blog_eval_examples.png)

*One example from each benchmark with the final model's answer. RefCOCO: "man with spectacles on the right". Where2Place: "the vacant space between the two mugs". Point-Bench: "Point to what the cows are pulling", where the point lands on the wagon.*

## Freeze the evals first

Before training anything I fixed three benchmarks, one output parser and one statistical test, and promised not to touch them again. Every number in this post comes from that frozen harness.

| Benchmark | Items | What it asks |
| --- | --- | --- |
| RefCOCO val | 2,000 | Point at the object a phrase describes ("man in red on the left") |
| Where2Place | 100 | Point at free space ("a spot between the mug and the plate") |
| PointArena Point-Bench | 979 | Open-world pointing across five categories: affordance, counting, reasoning, spatial and steerable |

A prediction counts if the first point lands inside the target mask (Point-Bench uses its own rule for counting). Decoding is greedy with a 64-token cap, and an unparseable answer is simply a miss.

Freezing early paid off on day one. The popular Hugging Face copy of RefCOCO "val" turned out to be more than half training data: only 3,811 of its 8,416 objects are in the real validation split. I sampled my 2,000 from those, rebuilt full masks from the raw COCO annotations, and filtered every training image against the eval images at the pixel level, not just by id.

The statistics matter as much as the data. Where2Place has only 100 items, so a 95% confidence interval is about ±10 points, and a 6-point gain can be noise. But two checkpoints are scored on the same items, which carries far more information. Call b the items only the new checkpoint gets right and c the items only the old one does. If the two are equally good, each of those disagreements is a coin flip:

<div class="math-display">
$$
b \sim \operatorname{Binomial}\big(b + c,\ \tfrac{1}{2}\big)
$$
</div>

That is the exact McNemar test. Items both models get right, or both get wrong, say nothing about the difference. With 18 new hits and 4 new misses, a 14-point gain is real (p = 0.004). With 4 and 8, a 4-point drop is noise (p = 0.39). Every comparison below uses this test.

## How far off is 2B?

Out of the box, Qwen3-VL-2B already points reasonably well, and scaling the same family to 32B adds about 7 points on RefCOCO and Point-Bench and 21 on Where2Place. That gap was the target.

| Out of the box | RefCOCO | Where2Place | Point-Bench | Latency (p50) |
| --- | --- | --- | --- | --- |
| Qwen3-VL-2B, my prompt | 80.6% | 40% | 48.9% | 274 ms |
| Qwen3-VL-2B, native format | 86.2% | 44% | 51.9% | 314 ms |
| Molmo-7B-D | 89.0% | 31% | 56.6% | 915 ms |
| Qwen3-VL-32B (AWQ 4-bit) | 93.2% | 65% | 59.0% | 885 ms |

Each baseline gets its best shot, through an adapter for its own output format. The 32B was quantized to 4 bits to fit the card and served with vLLM; the others ran in Hugging Face transformers.

Two details are worth noticing. The 2B loses 6 points on RefCOCO in my format purely from formatting slips: 126 answers closed the tag with `]`. And the 32B is weak at counting (15%). Asked to point at "all the apples" under an instruction that says "exactly one point", it obeys the instruction.

## SFT does most of the work

Supervised fine-tuning on 60,000 examples took the 2B from 80.6% to 91.1% on RefCOCO and from 48.9% to 58.6% on Point-Bench, already past Molmo-7B-D. It took 4.6 hours.

The data mixes three sources. PixMo-Points gives open-vocabulary, human-clicked points. RefCOCO, RefCOCO+ and RefCOCOg give referring expressions, from short ("left dog") to long and relational. RoboPoint gives tabletop objects and free-space regions. I trained a LoRA adapter (rank 16) on the language model for two epochs and left the vision encoder frozen.

One small decision mattered: how to turn a mask into a single target point. The centroid is the obvious choice and a bad one. For a crescent, a ring or a person with outstretched arms, it can fall outside the object. I used the deepest pixel of the mask instead, the point farthest from its boundary. It is always inside, and it leaves the most room for a few pixels of aim error.

Then I looked at what the SFT model still got wrong, and most misses were not aim errors. They were misreadings: ordinals ("second man in from left"), relations ("slice to left of lifter"), distance ("person sleeping far away") and typos ("rigth bike"). The point usually lands cleanly on an object, just the wrong one.

![Five RefCOCO misses by the SFT model](/images/pointing-part1/sft_refcoco_misses.png)

*SFT misses on RefCOCO. Green is the target, the red dot is the SFT model's point, and the white ring is the base model's.*

![SFT versus the base model: two wins and two failures](/images/pointing-part1/blog_sft.png)

*Green is the target, red the SFT point, and the white ring the base model's point.*

1. "Point to what could ignite things": the base model points at the flame, SFT at the lighter.
2. "A vacant area on the front portion of the stove": the base model lands on the floor, SFT on the free cooktop.
3. "2nd guy from the left": SFT points at the leftmost person.
4. "The building to the left of the rightmost building": SFT points at the rightmost building itself.

Misreadings looked like exactly the kind of thing RL should fix. The model can clearly see the objects. It just needs to be rewarded for picking the right one.

## GRPO in one page

GRPO (Group Relative Policy Optimization) is the RL algorithm behind most recent reasoning models. Stripped down, it is the oldest policy gradient method there is, REINFORCE, with one clever baseline.

The policy gradient says: sample an answer, and push up its log-probability in proportion to how good it was relative to a baseline b:

<div class="math-display">
$$
\nabla J = \mathbb{E}\big[(r - b)\,\nabla \log \pi_\theta(o \mid q)\big]
$$
</div>

The baseline can be anything that does not depend on the sampled answer, because the score function averages to zero: the sum over o of π·∇log π is ∇ of the sum of π, which is ∇1 = 0. A good baseline does not change the expected gradient, but it cuts the noise enormously. Without one, every answer to an easy prompt gets pushed up just for being on an easy prompt.

PPO learns the baseline with a second network. GRPO instead samples G answers to the same prompt and uses their average reward. That costs samples instead of memory, the right trade on a 24 GB card. Each answer's advantage is its reward standardized within its group:

<div class="math-display">
$$
A_i = \frac{r_i - \operatorname{mean}(r_{1..G})}{\operatorname{std}(r_{1..G})}
$$
</div>

With a 0/1 reward this has a neat reading. If k of G = 8 answers hit, a hit gets +√((1 − p)/p) and a miss gets −√(p/(1 − p)), where p = k/8. One hit out of eight earns +2.65; one miss out of eight costs −2.65. The rare win on a hard prompt is pushed hardest, and so is the rare slip on an easy one.

Two more pieces complete the picture. GRPO keeps PPO's clipped probability ratio, but I took one gradient step per batch of samples, so the policy that sampled is the policy being updated, the ratio is exactly 1, and clipping never fires. On-policy GRPO really is REINFORCE with a group baseline, and my logs showed a clip fraction of 0 on every step. That is a useful unit test: if it is not zero, your sampler and trainer disagree.

The second piece is a KL penalty that keeps the policy near the SFT model, because my main risk was forgetting RefCOCO. The penalty's effect has a closed form. Maximizing reward minus β times the KL is solved by:

<div class="math-display">
$$
\pi^*(o \mid q) \;\propto\; \pi_{\text{SFT}}(o \mid q)\; e^{\,r(q,o)/\beta}
$$
</div>

So β is a temperature on how far the reward may reweight what SFT already does. I used β = 0.02, and the reference model cost no memory: it is the same weights with the LoRA adapter switched off.

## Run 1: a gain on the wrong benchmark

The reward was as simple as it gets: 1 if the answer is exactly one well-formed point inside the target, otherwise 0. It calls the frozen eval's own parser and mask check, so on RefCOCO-style prompts the training signal is the metric itself. I deliberately kept format inside that 0/1 rather than adding a format bonus. A bonus pays well-formed misses, and once every answer is well formed, which happened at step 1, it is a constant that only adds noise.

The prompt pool was 28,000 items: spatial and ordinal phrases from the RefCOCO family, other RefCOCO prompts to guard against forgetting, and RoboPoint objects and free space. Each step sampled 16 prompts × 8 answers and took 11 seconds, so 600 steps took about two hours.

The first smoke test exposed the core problem with RL on a model that is already good. A group whose eight answers all get the same reward has zero advantage everywhere: a full rollout for no gradient. If the model solves a prompt with probability p, that happens with probability:

<div class="math-display">
$$
P(\text{all the same}) = p^{G} + (1-p)^{G}
$$
</div>

At G = 8 that is under 1% for p = 0.5 but 43% for p = 0.9, and my SFT model was 91% on RefCOCO-style prompts. So before training I sampled every pool prompt eight times with the SFT model and kept only those it got right sometimes but not always. 15,483 of the 28,000 survived. Even then, late in training most groups were unanimous again, because the model was learning them.

The result was a real gain in an unexpected place. Where2Place jumped from 51% to 65% (p = 0.004) and RefCOCO rose from 91.1% to 92.8%. But Point-Bench, the benchmark I cared most about, went from 58.6% to 57.5%. The free-space third of the pool matched Where2Place almost exactly. The "spatial" RefCOCO prompts I had hoped would fix misreadings were mostly easy left-or-right cases the model already solved.

![Run 1 versus SFT: two wins and two failures](/images/pointing-part1/blog_grpo_v1.png)

*Red is run 1, the white ring SFT.*

1. "The free region under the dresser": SFT sits on the dresser's edge; run 1 moves into the space underneath.
2. "The empty space to the right of the toy tower": SFT lands off the table; run 1 lands on the empty tabletop.
3. "The object to the left of the sink": SFT was right; run 1 lands on the bare counter.
4. "The object to the left of the pink vase": SFT was right; run 1 drifts onto the pink vase itself.

The two losses share a pattern that recurs later: the model lands on the anchor object named in the relation instead of the target. Many of the free-space wins, meanwhile, are near misses: SFT's point sits on an object's edge, and run 1 nudges it into the free space.

## Run 2: fixing errors the benchmark does not ask about

If the training prompts were too easy, I would make hard ones. COCO has a mask for every instance of its 80 object classes, so I could generate phrases with exact answers: "the second cup from the left", "the bottle to the left of the laptop", "the person nearest the table", plus typos. The hard part was ambiguity. Spot checks found "second person from the right" counting a speck in the far background, and "the person below the frisbee" when two people were below it. I added filters so that every counted instance is a fair size and every distractor fails even a loose reading of the relation.

On held-out images, run 2 learned exactly what it was taught. Relations went from 74% to 92% and ordinals from 42% to 53%. Point-Bench did not move at all: 58.7%.

Only then did I do what I should have done first, and sorted the SFT model's 242 Point-Bench misses by cause:

| Why it missed | Share |
| --- | --- |
| Reasoning and affordance ("the object that can produce light") | 41% |
| Refers to an "existing point" not shown in the image or prompt | 33% |
| Spatial: object parts, viewpoint, small targets | 22% |
| Other | 4% |

Ordinals appear in 5 of Point-Bench's 784 non-counting items. I had trained a skill the benchmark barely asks for. A third of the misses cannot be fixed by training in my harness, because the question refers to a point the model is never shown. The biggest fixable bucket was function and world knowledge.

![Run 2 versus SFT: two wins and two shared failures](/images/pointing-part1/blog_grpo_v2.png)

*Red is run 2, the white ring SFT.*

1. "2nd sink from the left": SFT picks the first sink; run 2 picks the second.
2. "The window to the right of the red door": SFT picks the window above the door; run 2 gets it right.
3. "The object the car is stopping for": both point at a parked bicycle, not the traffic light.
4. "The tool that creates music": both point at a shoulder strap, not the drum.

The first two are the skills run 2 taught. The last two are the kind of question Point-Bench actually asks, and run 2 did nothing for them.

## Run 3: the pass@8 check, and a behavior nobody paid for

Before spending another four hours, I asked a cheaper question: can RL even reach this skill? The test takes an hour. Sample the current model 8 times on held-out prompts of the new kind and count the hits. If most prompts are 0 out of 8, there is no reward to follow, and RL cannot help. If many are somewhere between 1 and 7, RL has something to sharpen.

For function-style prompts ("the utensil used to cut food"), the SFT model was right 83% of the time greedily and 93% of the time at least once in eight tries, with almost half the prompts in the learnable 1-to-7 band. Even better, 33 of its 48 greedy misses were correct when the object was simply named. The knowledge was there; the reading was unreliable. That is exactly what RL is good at.

Run 3 added 9,000 function prompts and one change to the reward. About a fifth of misses landed just outside the mask, so I gave partial credit for near misses, fading to zero at 3% of the image diagonal:

<div class="math-display">
$$
r = \begin{cases} 1 & \text{inside the mask} \\ 0.5 \cdot \max\!\big(0,\ 1 - d \,/\, (0.03 \cdot \text{diagonal})\big) & \text{otherwise} \end{cases}
$$
</div>

The radius is small on purpose: pointing at a neighboring object earns nothing.

Point-Bench rose to 59.9%, the best so far, but not for the reason I designed. The whole gain came from counting, which went from 16.4% to 21.5%; everything else was flat. On prompts like "Point to all the apples", run 3 had started listing every apple, even though the prompt says "exactly one point". Multi-point answers on Point-Bench had crept up run by run, from 5 after SFT to 46 after run 3. No reward had ever seen an "all the" prompt, so nothing held that behavior in place, and it drifted. It happened to help. It could just as easily have hurt.

![Run 3 versus SFT: wins and failures](/images/pointing-part1/blog_grpo_v3.png)

*Red is run 3, the white ring SFT.*

1. "The object people can use to paint": SFT picks a sponge stick; run 3 picks a paintbrush.
2. "The tool used to save pencils": SFT lands just above the pencil case; run 3 lands on it.
3. "Point to all the apples": the prompt asks for exactly one point, but run 3 gives three, one per apple, and is scored correct.
4. "The object in front of the black bus": SFT was right; run 3 points at the black bus itself, the anchor again.

## Run 4: when RL cannot start, SFT first

Counting was now the obvious target, so I made it explicit. PointArena counts an answer correct only if the number of points equals the number of objects and every object is hit. COCO's instance masks give exact counts, so the reward could apply that rule exactly.

The pass@8 check said no. On held-out "all the X" prompts with two or more objects, the SFT model was right 0 times out of 8 on COCO images. There was nothing for RL to reinforce.

So run 4 has two stages. A short SFT stage (5,800 COCO counting examples, 5,000 multi-point PixMo examples, and 8,000 single-object examples so the one-point habit survives) took held-out counting from 0% to 48% and moved half the prompts into the learnable band. GRPO then sharpened it with the counting rule as the reward, plus half credit scaled by how well the points matched the objects (their F1).

Counting on Point-Bench went from 21.5% to 30.3%, with 17 items gained and none lost. The multi-point answers stayed where they belong: 5 of 784 non-counting Point-Bench items and 4 of 2,000 RefCOCO items, none on Where2Place. Overall Point-Bench reached 60.9%, the first gain over SFT that clears p < 0.05. The cost was Where2Place, which slipped from run 3's 63% to 57% (not significant) because the counting stage diluted the free-space prompts.

![Run 4 versus run 3: wins and failures](/images/pointing-part1/blog_grpo_v4.png)

*Red is run 4, the white ring run 3.*

1. "Point to all the glasses of drinks": run 3 gives one point; run 4 gives one per glass, all three inside.
2. "Point to all the cars": run 3 gives one point; run 4 gives one per car.
3. "The item that people can drink": run 3 picks a dish; run 4 picks the glass.
4. "The free space above the drawers": run 3 was right; run 4 jumps to the top-right corner. This is the Where2Place cost.

## Where it ended up

The final 2B is statistically level with the 4-bit 32B on Point-Bench and RefCOCO, at a tenth of the latency.

![Point-Bench accuracy by training stage](/images/pointing-part1/pointbench_by_stage.png)

|  | RefCOCO | Where2Place | Point-Bench | Latency (p50, vLLM) |
| --- | --- | --- | --- | --- |
| Qwen3-VL-2B, after SFT + 4 GRPO runs | 92.1% | 57% | 60.9% | 90 ms |
| Qwen3-VL-32B (AWQ 4-bit), out of the box | 93.2% | 65% | 59.0% | 885 ms |

Neither difference is significant (Point-Bench p = 0.25, RefCOCO p = 0.12). On Where2Place the final model trails, but run 1 matches the 32B's 65%. Scores are from the frozen Hugging Face harness; served with vLLM under the 32B's settings, the 2B scores 92.2%, 62% and 61.5%. The 2B also needs about 4.5 times less memory for its weights.

The category split shows the two models win in different places:

| Point-Bench category | 2B after SFT | 2B final | 32B | Molmo-7B-D |
| --- | --- | --- | --- | --- |
| Affordance | 82.3% | 85.9% | 87.4% | 81.8% |
| Counting | 16.4% | 30.3% | 15.4% | 26.2% |
| Reasoning | 66.8% | 66.3% | 71.0% | 67.9% |
| Spatial | 72.8% | 70.3% | 75.9% | 66.2% |
| Steerable | 54.5% | 51.5% | 45.5% | 40.9% |

The 32B leads on reasoning and spatial questions, which need knowledge that scale provides and RL did not add. The 2B's edge comes almost entirely from counting: the one skill I put in with SFT and then sharpened with RL. RL closed the gap where it had something to sharpen, and only there.

## How this compares with published work

The recipe here is not new. What I think is worth sharing is how each stage was chosen and checked.

Pointing as an output format went mainstream with [Molmo and its PixMo-Points data](https://allenai.org/blog/molmopoint). [RoboPoint](https://arxiv.org/abs/2406.10721) brought it to robotics, and Where2Place comes from that line. [PointArena](https://arxiv.org/abs/2505.09990) built Point-Bench and found that explicit pointing supervision matters far more than model size, and that chain-of-thought does not help.

SFT followed by GRPO with a hit-the-region reward is now the standard recipe:

- [Embodied-R1](https://arxiv.org/abs/2508.13998) trains a 3B model with in-mask and distance rewards and reaches 69.5% on Where2Place.
- [Poivre](https://arxiv.org/abs/2509.23746) has a 7B model predict a point, see it drawn on the image, and refine it. It is trained with RL alone on 8,000 PixMo examples and reaches 67.5% on Point-Bench, above Molmo-72B.
- [RoboRefer](https://arxiv.org/abs/2506.04308) and GUI click-grounding models apply the same idea to spatial references and screen elements.

The best published Point-Bench scores sit around 68–71% for 7–8B models, so 60.9% at 2B is respectable but not state of the art. The numbers are also not directly comparable: my frozen prompt, token cap and parser differ from each paper's protocol. Poivre, for example, reports 61.6% for Molmo-7B-D, where my harness gives 56.6%.

What this project adds is the process. Pass@8 decided whether a skill needed SFT or only RL. Synthetic data targeted measured errors, with explicit rules against ambiguous labels. And every claim was a paired significance test, including the negative ones, such as run 2's skills not transferring and run 3's gain being pure drift. Most papers report only the final average.

## What I would tell myself at the start

1. **Freeze the evals, and check what "val" means.** Every argument here rests on comparing checkpoints on identical, leak-checked items.
2. **Study the benchmark's misses before designing data.** Use the eval to choose what to train, never to tune on its items.
3. **Run pass@8 before every RL run.** Hits in the 1-to-7 range mean RL can work. Zero means you need SFT data first. This one-hour check would have predicted all four runs.
4. **Reward exactly what you score, and watch what you don't.** Behavior the reward never sees will drift.

5. **Use paired tests.** On 100 items, a 6-point swing can be noise and a 14-point one is not.

Part 2 picks up from here: winning back Where2Place without losing counting, quantizing the 2B for on-device use, and porting the recipe to a smaller on-device model.

## Appendix: bugs that train without errors

Each of these trains happily and quietly computes the wrong gradient.

| Trap | What goes wrong | Fix |
| --- | --- | --- |
| LoRA dropout above 0 | Sampling runs without dropout and scoring with it, so the probability ratio is never 1 | Set LoRA dropout to 0 |
| The checkpoint's sampling defaults | Qwen3-VL's generation config sets temperature 0.7, top-k 20 and top-p 0.8, so samples come from a truncated distribution while log-probs use the full one | Pass temperature, top-p 1 and top-k 0 explicitly, and divide logits by the same temperature when scoring |
| Decoding, then re-encoding | Tokenization does not round-trip, so you score tokens you never sampled | Append the sampled token ids directly |
| Right padding in batched generation | Rows predict from different positions | Left-pad to generate, right-pad to score |
| Per-token side inputs | Qwen3-VL's token-type ids stayed at prompt length, which broke its position encoding | Extend every per-token tensor over the answer |
| Loss normalized per micro-batch | The gradient depends on how sequences are packed | Normalize by the token count of the whole step |

Training settings for all four runs: LoRA rank 16, G = 8, 16 prompts per step, 600 steps, learning rate 2e-5, clip range 0.2/0.28, KL weight 0.02 with the k3 estimator, sampling at temperature 1.
