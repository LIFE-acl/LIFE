# The paper is under submission to the 2025 Association for Computational Linguistics.

Pytorch code for paper: **"Prompt-Induced Linguistic Fingerprints: Decoding Reconstruction Probability Shifts for LLM-Generated Fake News Early Detection"**

---

# 📌 Overview

This directory contains code necessary to run **LIFE**, a LLM-generated fake news detection architecture. See our paper for details on the code.

## 📌 Model Framework
<div align="center">
    <img src="Figures/figure1.png" alt="Model Framework" width="60%">
</div>

## 📌 Probability Shifts
<div align="center">
    <img src="Figures/figure2.png" alt="Probability Shifts" width="60%">
</div>

---

# 📚 Dataset

Our dataset includes **Politifact_Llama, Gossipcop_Llama, and FakeLLM**. We will provide these datasets under this directory.

Additionally, the generation methods for **Politifact_Llama and Gossipcop_Llama** are from the following project:

🔗 [Method in creating Politifact_Llama and Gossipcop_Llama](https://github.com/mbzuai-nlp/Fakenews-dataset)

The **FakeLLM** dataset can be obtained from the following project:

🔗 [FakeLLM Datasets](https://github.com/llm-misinformation/llm-misinformation/)

---

# ⚙️ Requirements

It is recommended to create a **Conda virtual environment** to run the code.  
The Python version is **3.8.0**. The detailed versions of required packages are available in `requirements.txt`.

📌 Install all the required packages using:

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Code

1. **Extract key sentences**  
   ```bash
   python KeySentenseExtraction.py
   ```
2. **Obtain key sentence reconstruction probability vector**  
   ```bash
   python keySentenceProb.py
   ```
3. **Train the model**  
   ```bash
   python train.py
   ```

---