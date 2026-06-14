+++
date = '2026-06-13T21:13:20-07:00'
draft = true
title = 'World Model Essay - Dreamer Series'
tags = ["machine-learning", "world-models"]
+++

Introduction (delete the sub title though)





What is this about, 



Higher level architecture

The initial architecture for learning the world model was published in 2018 by David Ha [1]. 



The dreamer makes it fully differentiable and scalable by replacing RNN with RSSM, integrating latent space with history and….

## Dreamer Core - RSSM

One of the core contributions of the Dreamer paper is the Recurrent State Space Model. It computes two latents, a deterministic one ($h_t$) and a stochastic one ($z_t$). The deterministic latent is a recurrent state, which tells us how the world has evolved. The stochastic latent tells us the model’s current belief about the world. 

The recurrent state ($h_t$) is modelled using the previous recurrent state, previous action, and the previous stochastic latent. 
$$h_t = f(h_{t-1}, z_{t-1}, a_{t-1})$$
This acts like a memory, accumulating the events across time and tracking how the world has evolved. 
The stochastic latent ($z_t$) is the model’s belief of the world right now. It is modelled as a distribution of possible states rather than a deterministic state, as the world is partially observable and the future is probabilistic. The distribution can be computed in two ways:
- When the observation (ot)is available, the model infers from both the recurrent state and the observation. 
- When no observation is available, the model infers from the recurrent state alone. 

The two ways of computing the stochastic vector are crucial, as this is the main reason we can learn from imagination. 

The beautiful part is that we can operate this RSSM in 3 ways, making it easier for us to train on real data, learn while dreaming, and deploy it. Let’s explore each mode:
- Learning the world
- Dreaming
- Deployment

## Learning the World

This is the first stage of training. Given a dataset of (observation, action, and reward), we train the model to learn the world dynamics. The RSSM gets the following inputs
- Previous recurrent state ($h_{t-1}$)
- Previous stochastic state ($z_{t-1}$)
- Previous action ($a_{t-1}$)
- Encoded observation ($o_t$)

The current recurrent state ($h_t$) is computed as follows
$$h_t = f(h_{t-1}, z_{t-1}, a_{t-1})$$

We can compute the stochastic latent's posterior from current recurrent state and the observation.
$$\mu_{z},\sigma_{z} = f(h_t, o_t)$$ 

Then $z_t$ is can be sampled from $\mu_{z},\sigma_{z}$.

Now, we want to train the model to dream, i.e. get the model's belief without external observation ($o_t$). To make this happen, we would want to compute the prior distribution from $h_t$. 
$$\mu_{z},\sigma_{z} = f(h_t)$$

Reward is also computed.

To train this model, we attach a decoder to reconstruct the observation based on $z_t$. It is optimized for reconstruction loss and KL divergance between prior and posterior distribution of the stocahstic latent $z_t$.


All the Math



Evolution



How to Use



Reference

[1] 



Notes





You wrote:



“thus it keeps track of state of the period of time.”

This is true but underspecified. The power of the recurrent state ht  h_t  ht​ is that it provides a stable, deterministic backbone that can propagate information over long horizons without vanishing gradients. The stochastic zt  z_t  zt​ then only needs to model local uncertainty around this backbone. This geometric decomposition (stable manifold + stochastic deviation) is what allows RSSM to scale to longer horizons than Ha’s MDN-RNN.



