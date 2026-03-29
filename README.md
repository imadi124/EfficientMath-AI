# 🧮 EfficientMath-AI: Full-Stack LLM Web Application

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Unsloth%20%7C%20Transformers-orange)
![Format](https://img.shields.io/badge/Format-GGUF%20(4--bit)-lightgrey)
![Deployment](https://img.shields.io/badge/Deployed-Hugging%20Face%20Spaces-yellow)

> EfficientMath-AI is a parameter-efficient fine-tuned (PEFT) version of Meta's **Llama-3.1-8B**, explicitly optimized to solve multi-step grade school math problems using CoT(Chain of Thought) reasoning. By leveraging PEFT(Parameter-Efficient-Fine-Tuning), LoRA((Low-Rank Adaptation), 4-bit GGUF quantization, Gradio, colab T4 GPU and a custom C-library cloud deployment architecture, this project bridges the gap between raw data science and full-stack software engineering.

**[Launch the Live Web App Here](https://huggingface.co/spaces/iamabhayaditya/EfficientMath-AI)**

<img width="1893" height="522" alt="image" src="https://github.com/user-attachments/assets/62434392-b3ce-408a-81d0-a5f621559344" />

---

## 🏗️ Project Architecture & Pipeline

This repository documents the entire machine learning lifecycle, from data curation to local hardware inference.

1. **Data Curation:** Formatted the `openai/gsm8k` dataset to enforce strict `### Problem:` and `### Solution:` sequential reasoning.
2. **PEFT Fine-Tuning (LoRA):** Utilized the **Unsloth** framework on a free-tier NVIDIA T4 GPU to train the 8-billion parameter model efficiently (Rank = 16, Alpha = 16).
3. **Hardware Democratization (GGUF):** Merged the base model with the custom LoRA adapters and aggressively quantized the architecture to 4-bit integers (`Q4_K_M`) using `llama.cpp`. This compressed the model from ~16GB to 4.92GB.
4. **Cloud Deployment:** Built a responsive, streaming UI using **Gradio** and deployed it to Hugging Face Spaces. Handled deep Linux dependency conflicts (Alpine `musl` vs. Debian `glibc`) by configuring a direct pre-compiled `llama-cpp-python` wheel for the free CPU tier.
5. **Local Inference:** Configured the model for 100% offline, private use on standard consumer laptops via GPT4All.

---

## 📊 Evaluation & Metrics

The model was strictly evaluated against a test split of the GSM8K dataset, focusing on numeric accuracy and reasoning coherence.

* **Overall Accuracy:** 66.0%
* **Perplexity:** Maintained a tightly clustered distribution (2.5 - 4.0), indicating high model confidence and fluency in mathematical syntax.
* **Verbosity:** Aligned closely with ground-truth human answers, avoiding the common LLM trap of hallucinating verbose, unnecessary text.

![research_graphs](https://github.com/user-attachments/assets/93c6b703-668a-469c-ab60-548a0947f656)

---

## 💻 How to Run Locally

You can run this model entirely offline on a standard laptop CPU. No expensive GPU is required.

### Option 1: The UI Method (GPT4All)
1. Download [GPT4All](https://gpt4all.io/).
2. Download the `Meta-Llama-3.1-8B.Q4_K_M.gguf` file from the [Hugging Face Model Card](https://huggingface.co/iamabhayaditya/EfficientMath-AI).
3. Move the file into your local GPT4All models directory (e.g., `C:\Users\YOUR_NAME\AppData\Local\nomic.ai\GPT4All\`).
4. **Important Settings:** * In GPT4All Settings -> Models, set **GPU Layers to 0** to rely entirely on your CPU.
   * Paste the following into the **Chat Template** box:
```jinja
   {% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{'<|start_header_id|>assistant<|end_header_id|>\n\n'}}{% endif %}
```

### Option 2: Python / Backend Integration
For developers looking to integrate the math engine into their own applications:
```
pip install llama-cpp-python
```
Python
```
from llama_cpp import Llama

# Load the locally downloaded GGUF model
llm = Llama(
    model_path="./Meta-Llama-3.1-8B.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4  # Adjust based on your CPU cores
)

prompt = """Below is a math word problem. Solve it step by step and provide the final answer.

### Problem:
Abhay has 16 apples, he borrowed 5 from Akash then gave 14 to Shivam. How many apples is he left with?

### Solution:
"""

output = llm(
    prompt,
    max_tokens=256,
    temperature=0.2,
    stop=["<|eot_id|>"]
)

print(output["choices"][0]["text"])
```

📂 Repository Structure
```
EfficientMath-AI/
├── README.md
├── LICENSE ← MIT License
├── .gitignore
├── requirements.txt
├── notebooks/
│ ├── 01_fine_tuning.ipynb: The complete Unsloth and LoRA training pipeline.
│ ├── 02_gguf_quantization_gradio.ipynb: Post-training quantization, testing, and Hub deployment.
│
├── assets/ ← graph

```
## 🛠️ Contribution
Contributions to Efficient-AI are highly encouraged. Whether it is suggestions, improvements, bug fixes, documentation updates, or feature enhancements, feel free to contribute to the project repository.

## 📜 License
Efficient-AI is licensed under the MIT License, granting you the freedom to use, modify, and distribute the code in accordance with the terms of the license.
