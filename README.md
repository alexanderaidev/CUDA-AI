# CUDA-AI
A reinforcement learning approach on an AI that predicts blocksize and tilesize for a CUDA kernel.


# **⚠️ Note:**  
  This project is a research project and serves as a proof of concept, that we can use AI to program the hardware itself. 
  This project is not meant to break any current performances. The CUDA kernel is written by myself and minimalistic. 
  It is in constant development and therefore new files and also graphs will be added. 

# 1: Special characteristics
  - since the Reward for this AI is non-differentiable, we cannot use PyTorch or any other Framework which uses Autograd.
  - This AI agent ist buit with CuPy. It is allowing us to handle the forward and backward pass manually. - Therefore I built a framework-like construkt.
  - Also there currently is a declaration called 'optimizer' in the code and the optimizer is integrated inside of the model-class itself. But the current setup is not using the optimizer.
  - For optimization we currently use the typical gradient decent.
  - A new metrik is introduced and used.

# 2: The Model
  - The commented-out code is kept, so you could debug the model if you want to, until the project is finished. The final code will not include any commented-out code.
  - The agent is built with dense-layers only.

# 3: Training
  - The agent is trained by a mixture of exploration an exploitation.
  - The agent is trained with the exponential moving average (ema).
  - The EMA relies solely on the actual model performance, rather than the exploration performance. The metric outlined above is used as the basis for this evaluation.

# 4: Metrik
  - LOG10(Floating point operations per second (FLOPs/s) divided by Floating point operations power to alpha (FLOPs^alpha))
