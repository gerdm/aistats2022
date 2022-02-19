---
# try also 'default' to start simple
theme: default
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://images.unsplash.com/photo-1620460700571-320445215efb?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&h=1080&ixid=MnwxfDB8MXxyYW5kb218MHw5NDczNDU2Nnx8fHx8fHwxNjQ1MjY3NTc2&ixlib=rb-1.2.1&q=80&utm_campaign=api-credit&utm_medium=referral&utm_source=unsplash_source&w=1920
# apply any windi css classes to the current slide
---

# Subspace Neural Bandits
### AIStats 2022

Gerardo Duran-Martin, Queen Mary University of London, UK  
Aleyna Kara, Boğaziçi University, Turkey  
Kevin Murphy, Google Research, Brain Team

Feburary 2022

----

# Contextual bandits
## Li, et.al. (2012)

Let $t=1,\ldots,T$. At every time step $t$,

1. we are given a set of action $\mathcal{A} = \{a^{(1)}, \ldots, a^{(K)}\}$ to choose from;
2. we are given a context ${\bf s}_t$ to decide on an action to take
3. we decide (based on some algorithm) some action $a_t \in \mathcal{A}$ to take
4. we obtain a reward $r_t$ based on the context ${\bf s}_t$ and the action $a_t$

<span style="background-color:#A7C7E7">Our goal is to maximise the expected reward $\sum_{t=1}^T\mathbb{E}[R_t]$</span>

---

# Thompson Sampling
### One way to solve the bandit problem.

Let $\mathcal{D}_t = (s_t, a_t, r_t)$ be a sequence of observations. Let $\mathcal{D}_{1:t} = \{\mathcal{D}_1, \ldots, \mathcal{D}_t\}$. Then, at every $t=1, \ldots, T$:

1. Sample $\boldsymbol\theta_t \sim p(\cdot \vert \mathcal{D}_{1:t})$
2. $a_t = \arg\max_{a \in \mathcal{A}} \mathbb{E}[R(s_t,a; \boldsymbol\theta_t)]$
3. Obtain $r_t \sim R(s_t,a_t; \boldsymbol\theta_t)$
4. Store $\mathcal{D}_t = (s_t, a_t, r_t)$

<!-- ToDo: Add bayesian-linear-regression bandit example in pyprobml-->

---

# Neural Bandits
### Characterising the reward function

Let $f: \mathcal{S}\times\mathcal{A}\times\mathbb{R}^D \to \mathbb{R}^K$ be a neural network. A neural bandit is a contextual bandit where the reward is taken to be

$$
  p(r_t \vert {\bf s}_t, a, \theta_t) = \mathcal{N}\Big(r_t \vert f({\bf s}_t, a, \boldsymbol\theta_t), \sigma^2\Big)
$$

The main question: <span style="background-color:#A7C7E7"> How to determine $\boldsymbol\theta_t$ at every time step $t$ using Thompson sampling?</span>  

We need to compute (or approximate) the posterior distribution of the parameters in the neural network:
$$
    p(\boldsymbol\theta \vert \mathcal{D}_{1:t}) \propto p(\boldsymbol\theta) p(\mathcal{D}_{1:t} \vert \boldsymbol\theta)
$$


---
layout: two-cols
---
# Partially Bayesian
* Neural linear approximation
* Lim2 approximation
* Neural tangent approximation

::right::

# Fully Bayesian
* HMC (Hamiltonian Monte Carlo) sampling of posterior $p(\boldsymbol\theta \vert \mathcal{D}_{1:t})$
* Extended Kalman filter (EKF) online estimation of Neural Bandits
* EKF subspace online estimation of Neural Bandits (Our method)