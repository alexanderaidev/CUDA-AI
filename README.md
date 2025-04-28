# CUDA-AI

A reinforcement learning approach for training an AI to predict optimal block size and tile size configurations for a CUDA kernel.

---

## ⚠️ Note

This project is a research initiative and serves as a **proof of concept** demonstrating that AI can be utilized to program hardware directly.  
It is **not intended to outperform existing state-of-the-art CUDA libraries**.  
The CUDA kernel used in this project was written manually and is intentionally minimalistic to facilitate experimentation.  
The project is under **active development**, and new files, results, and graphs will be added continuously.
The comment language aswell as the debug language (print-command) will be switched to english in the next update.

---

## 1. Special Characteristics

- Since the reward function in this project is **non-differentiable**, frameworks relying on automatic differentiation (e.g., PyTorch, TensorFlow) cannot be utilized.
- The AI agent is built entirely using **CuPy**, allowing for **manual control over the forward and backward passes**.  
  Consequently, a **framework-like structure** was developed manually.
- There is currently a module named **"optimizer"** declared in the code, and an optimizer framework is integrated into the model class.  
  However, the current implementation **does not yet utilize** this optimizer during training.
- Optimization is currently performed using **standard gradient descent**.
- A **novel metric** has been introduced and is employed for model evaluation.

---

## 2. The Model

- **Commented-out code** has been deliberately retained during development to facilitate debugging and experimentation.  
  It will be **removed** in the final version of the project.
- The agent architecture consists exclusively of **dense layers**.

---

## 3. Training

- The agent is trained through a mixture of **exploration** and **exploitation** strategies.
- Training leverages an **Exponential Moving Average (EMA)** of the model's performance.
- The **EMA** is based solely on the **actual model performance**, rather than the **exploration performance**.  
  The **metric** described below is used as the basis for EMA computation.

---

## 4. Metric

The model's performance is evaluated using the following novel metric:

Metric = log10(FLOPs/s divided by FLOPs^alpha)

where:
- **FLOPs/s** = floating point operations per second (measured runtime performance),
- **FLOPs** = theoretical number of floating point operations,
- **α** (alpha) = scaling factor introduced to normalize performance across varying matrix sizes.

---
