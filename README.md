# 🔐 Watermarking Language Models: Red List / Green List vs. Cluster Watermarking

This repository contains the code and results of a comparative study on **text watermarking methods** applied to transformer-based language models: **OPT-350M** and **GPT-2 Medium**. We evaluate and contrast two prominent techniques—**Red List / Green List** watermarking and **Cluster watermarking**—across standard NLP evaluation metrics such as **BLEU**, **ROUGE**, and **Perplexity**.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Models Used](#models-used)
- [Watermarking Techniques](#watermarking-techniques)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Skills Used](#skills-used)
- [How to Run](#how-to-run)
- [References](#references)

---

## 🧠 Introduction

With the increasing use of large language models (LLMs), watermarking techniques have emerged as a tool to attribute and verify generated content. In this study, we focus on **evaluating the trade-offs** between two watermarking methods on **fluency**, **semantic preservation**, and **generation quality**. 

---

## 🏗️ Models Used

- **OPT-350M** – Open Pre-trained Transformer by Meta.
- **GPT-2 Medium** – 345M parameter model released by OpenAI.

These models were chosen to analyze watermarking effects across different architectures and parameter scales.

---

## 🛠️ Watermarking Techniques

### 1. Red List / Green List Watermarking
- Adds constraints on token selection using predefined "green" (preferred) and "red" (discouraged) token sets.
- Tuned using hyperparameters:
  - **Gamma** (controls strictness of green list adherence)
  - **Delta** (margin between list probabilities)

### 2. Cluster Watermarking
- Applies clustering on the vocabulary using k-means and selects tokens from clusters to impose watermarks.
- Emphasizes distributional control without significantly altering generation quality.

---

## 📏 Evaluation Metrics

We used three key evaluation metrics:

- **BLEU**: Measures n-gram overlap (fluency and faithfulness)
- **ROUGE**: Measures recall-oriented overlap (summarization)
- **Perplexity**: Measures coherence and naturalness of generated text

---

## 🧪 Experimental Setup

- For each model and watermarking method:
  - Conducted generation on a fixed prompt dataset
  - Evaluated across multiple hyperparameter configurations:
    - **Gamma** values: 0.3, 0.5, 0.6
    - **Delta** values: 1.0, 1.5, 2.0

- Tools used:
  - Python (PyTorch, HuggingFace Transformers)
  - Scikit-learn (for clustering)
  - NLTK & SacreBLEU (for evaluation)

---

## 📊 Results

### OPT-350M:
- **Red List / Green List** watermarking showed superior **BLEU** scores, particularly at **Gamma = 0.6** and **Delta = 2.0**
- **Cluster watermarking** performed better in terms of **Perplexity**, implying higher coherence.

### GPT-2 Medium:
- **Red List / Green List** achieved better **ROUGE-1** scores, notably at **Gamma = 0.5**
- **Cluster watermarking** again yielded **lowest perplexity**, indicating better quality and coherence.

| Model         | Method               | Best BLEU | Best ROUGE | Lowest Perplexity |
|---------------|----------------------|-----------|------------|--------------------|
| OPT-350M      | Red/Green List       | ✅         | ✅          | ❌                  |
|               | Cluster Watermarking | ❌         | ❌          | ✅                  |
| GPT-2 Medium  | Red/Green List       | ✅         | ✅          | ❌                  |
|               | Cluster Watermarking | ❌         | ❌          | ✅                  |

---

## ✅ Conclusion

- **Cluster watermarking** emerges as the superior method for ensuring **coherence and generation quality**, especially on **GPT-2 Medium**.
- **Red List / Green List** is more effective for tasks that demand **semantic fluency and fidelity**, especially on **OPT-350M**.
- Selection of a watermarking method should depend on the **model architecture** and the **desired trade-offs** in generation metrics.

---

## 🚀 Future Work

- Test watermarking methods on **larger models** (e.g., GPT-J, LLaMA-7B)
- Explore **adversarial robustness** and **detectability** of watermarked text
- Evaluate performance on **real-world downstream tasks** (summarization, translation)
- Combine watermarking with **reinforcement learning** to fine-tune generations

---

## 🧩 Skills Used

- 🤖 Natural Language Processing (NLP)
- 🧠 Deep Learning with Transformers
- 📊 Model Evaluation & Metric Design
- 🔐 Text Watermarking Algorithms
- 🧪 Experimental Design and Analysis

---

## 🛠️ How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/watermarking-llms.git
cd watermarking-llms

# Install requirements
pip install -r requirements.txt

# Run evaluation
python evaluate.py --model gpt2 --method redgreen --gamma 0.5 --delta 1.5
