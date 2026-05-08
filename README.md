MicroGPT Assignment

This project is based on Karpathy’s microGPT, which is a simple implementation of a GPT-style language model written in pure Python. In this assignment, I extended the original model by adding several modern techniques used in transformer models, including GELU activation, RoPE, LoRA, and Mixture of Experts (MoE).

GELU is an activation function that is commonly used in modern transformer models. Compared to ReLU, GELU is smoother and allows the model to learn more complex patterns. In this project, I implemented the GELU function inside the Value class and replaced the original ReLU activation in the MLP block with GELU.

RoPE (Rotary Position Embedding) is a method for adding positional information into the attention mechanism. Instead of using fixed positional embeddings, RoPE rotates the query and key vectors so that the model can better capture the relationships between tokens. In my implementation, I added RoPE functions and applied them to the query and key vectors before computing attention.

LoRA (Low-Rank Adaptation) is a technique that reduces the number of parameters that need to be trained by introducing low-rank matrices. Instead of updating the full weight matrix, LoRA adds a small low-rank update to it. In this project, I implemented a lora_linear function and applied it to the query projection in the attention layer.

Mixture of Experts (MoE) replaces a single feedforward network with multiple smaller networks called experts. A gating function is used to decide how much each expert contributes to the final output. In this implementation, I replaced the original MLP block with an MoE block, created multiple experts, and combined their outputs using a simple gating mechanism.

How to Run

To run the program, use the following command in the terminal:

python microgpt.py

The model will train on the dataset and then generate some sample outputs.

This project focuses on understanding the core ideas of these techniques rather than optimizing performance. All components are implemented manually in Python without using external libraries.

References

GELU: https://arxiv.org/abs/1606.08415
LoRA: https://arxiv.org/abs/2106.09685
RoPE: https://arxiv.org/abs/2104.09864
