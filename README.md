
# REINFORCE Variants for MNIST Classification

This project implements and compares variants of the REINFORCE algorithm from the seminal paper _"Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"_ (Williams, 1992), applied to a custom Gym-style environment for MNIST binary classification.

## 🧠 Objective

Apply and evaluate REINFORCE policy gradient methods in a supervised learning task, simulating reinforcement learning through episodic reward feedback in a classification setting.

## ⚙️ Environment & Dependencies

- Python 3.8+
- PyTorch
- OpenAI Gym
- torchvision
- matplotlib
- scikit-learn

Install dependencies:
```bash
pip install torch torchvision gym matplotlib scikit-learn
```

## 🚀 How to Run

1. Launch the Jupyter notebook:
   ```bash
   jupyter notebook MNIST_Classification_RL_Implementation_1000_Episodes.ipynb
   ```

2. Run all cells to train the agent using REINFORCE variants on a binary MNIST classification environment.

## 🧪 Variants Implemented

- **Basic REINFORCE**: reward-weighted gradient updates.
- **REINFORCE with Baseline**: uses running average to reduce variance.
- **Episodic REINFORCE**: updates weights after full episode.
- **Bernoulli Policy**: stochastic binary classifier.
- **Gaussian Policy**: continuous-valued output with learnable mean and std.

## 📈 Visualizations

- Reward trends (with optional exponential smoothing)
- Gradient variance tracking per variant
- Convergence curves by variant and policy type

## 📊 Results Summary

| Variant Type       | Policy     | Stability      | Reward Variance | Gradient Variance |
|--------------------|------------|----------------|------------------|--------------------|
| Basic              | Bernoulli  | Moderate       | Medium           | Medium             |
| Baseline           | Bernoulli  | **Most Stable**| Low              | **Low**            |
| Episodic           | Bernoulli  | Less Stable    | Low              |   Low              |
| Basic              | Gaussian   | Unstable       | High             | High               |
| Episodic           | Gaussian   | **Most Volatile** | **Very High** | **Very High**      |

## 📝 Notes

- All randomness is isolated to the policy output; the rest of the network remains deterministic.
- Uses the log-derivative trick to backpropagate through stochastic output.
- Actions correspond to class predictions, with reward based on classification accuracy.

## 📚 Reference

Ronald J. Williams (1992). _Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning_. Machine Learning.

## 🙌 Acknowledgements

Inspired by foundational reinforcement learning research and built using PyTorch and OpenAI Gym.
