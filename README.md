# IPO: Your Language Model is Secretly a Preference Classifier

Code and data for the paper "IPO: Your Language Model is Secretly a Preference Classifier"

Paper: https://arxiv.org/abs/2502.16182v1

Authors: Shivank Garg $^{1,2}$, Ayush Singh $^{1,2}$, Shweta Singh $^{1,3}$, Paras Chopra $^{2}$

$^1$ Indian Institute of Technology, Roorkee, $^2$ [Lossfunk](https://lossfunk.com/)

## Abstract: 
Reinforcement learning from human feedback (RLHF) has emerged as the primary method for aligning large language models (LLMs) with human preferences. While it enables LLMs to achieve human-level alignment, it often incurs significant computational and financial costs due to its reliance on training external reward models or human-labeled preferences. In this work, we propose Implicit Preference Optimization (IPO), an alternative approach that leverages generative LLMs as preference classifiers, thereby reducing the dependence on external human feedback or reward models to obtain preferences. We conduct a comprehensive evaluation on the preference classification ability of LLMs using RewardBench, assessing models across different sizes, architectures, and training levels to validate our hypothesis. Furthermore, we investigate the self-improvement capabilities of LLMs by generating multiple responses for a given instruction and employing the model itself as a preference classifier for Direct Preference Optimization (DPO)-based training. Our findings demonstrate that models trained through IPO achieve performance comparable to those utilizing state-of-the-art reward models for obtaining preferences.

![Model Diagram](assets/model_diagram.jpeg)

## Code for Reward Bench and RM Bench Evaluations 

- `binary_gpt.py:` About code and how to run

## Code for DPO Training
 
Steps and Each code usage

## Acknowledgements

The computational resources needed for the project were funded by [Modal Cloud](https://modal.com/) and [E2E Networks](https://www.e2enetworks.com/)
