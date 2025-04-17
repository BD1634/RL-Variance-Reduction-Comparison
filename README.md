# MNIST Classification with REINFORCE Algorithm (1000 Episodes)

This project reframes the MNIST digit classification task as a **reinforcement learning** problem. Using the REINFORCE algorithm introduced by Williams (1992), we evaluate several algorithmic variants to explore their learning dynamics and classification accuracy.

---

## 🧠 Project Highlights

- Implements the REINFORCE algorithm from:
  *Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229–256.*
- Evaluates multiple REINFORCE variants:
  - Basic REINFORCE
  - REINFORCE with baseline
  - Episodic REINFORCE
- Compares Bernoulli vs Gaussian policy distributions
- Tracks training reward and gradient variance for each method

---

## 📐 Methodology

We implement REINFORCE, a policy gradient method tailored for neural networks with stochastic output units. The approach adapts network weights solely through reward signals, without relying on explicit gradient estimates.

Each stochastic unit samples from a parameterized distribution:
\[
Y_i \sim g_i(\cdot \mid \mathbf{w}_i, \mathbf{x}_i)
\]
For a Bernoulli-logistic policy:
\[
P(Y_i = 1 \mid \mathbf{w}_i, \mathbf{x}_i) = \sigma(s_i), \quad s_i = \mathbf{w}_i^\top \mathbf{x}_i
\]
The objective is to maximize:
\[
J(\mathbf{W}) = \mathbb{E}_{\pi_\mathbf{W}}[r]
\]

Using the **log-derivative trick**:
\[
\nabla_\theta \mathbb{E}_{Y \sim p_\theta}[r(Y)] = \mathbb{E}_{Y \sim p_\theta}[r(Y) \nabla_\theta \log p_\theta(Y)]
\]
the REINFORCE update rule becomes:
\[
\Delta w_{ij} = \alpha (r - b_{ij}) \frac{\partial \log g_i(Y_i \mid \mathbf{w}_i, \mathbf{x}_i)}{\partial w_{ij}}
\]
For Bernoulli-logistic units:
\[
\frac{\partial \log g_i}{\partial w_{ij}} = (Y_i - P_i)x_j
\]

For episodic tasks:
\[
\Delta w_{ij} = \alpha (r - b_{ij}) \sum_{t=1}^{k} (Y_i(t) - P_i(t)) x_j(t-1)
\]

For Gaussian policy units:
\[
Y \sim \mathcal{N}(\mu, \sigma^2), \quad
\Delta \mu = \alpha (r - b) \frac{Y - \mu}{\sigma^2}, \quad
\Delta \sigma = \alpha (r - b) \frac{(Y - \mu)^2 - \sigma^2}{\sigma^3}
\]

Randomness is confined to the output layer, while deterministic hidden layers enable stable gradient flow via backpropagation.

---

## 📁 File Structure

