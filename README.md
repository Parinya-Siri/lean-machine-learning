# Dissertation Codebase: Language Models for Mathematical Theorem Proving

This repository contains the code, models, and training data associated with the experiments conducted for my dissertation research. The primary goal of this project is to investigate the capability of neural language models—specifically a fully connected neural network (FCN) and a GPT-2–based transformer model—to predict the next line of a Lean proof given several preceding lines. By comparing these two architectures, we aim to understand their relative strengths and weaknesses in modeling formal mathematical language and structure.

## Overview

**Research Objective:**  
The main objective of this experiment is to predict the next line of Lean code—taken from the Lean Mathematical Library (Mathlib)—using two distinct approaches:

1. **Fully Connected Neural Network (FCN):**  
   A baseline architecture that takes Unicode-encoded lines of Lean code as input.
   
2. **GPT-2 (Transformer-based) Model (GPT2-lean):**  
   A transformer-based architecture leveraging byte-level byte-pair encoding (BPE), trained to generate the next line of proof code.

**High-Level Findings:**  
- **FCN Model:**  
  Despite tuning hyperparameters and experimenting with various layer sizes, the FCN struggled to produce coherent next-line predictions. The accuracy remained near 0.0, and the model failed to learn meaningful patterns from Unicode representations of tokens.  
- **GPT2-lean Model:**  
  Although the accuracy in strictly matching the target line was also near 0.0, the GPT2-lean model generated more syntactically and semantically plausible lines of code. Its outputs were more readable and reflected the structural patterns of Lean proofs.

## Data

**Mathlib (Lean Mathematical Library):**  
We utilize Lean’s Mathlib, a comprehensive library of formally verified mathematics, which as of August 2022 contained about:

- **2,513 Lean files**
- **~460,000 lines of code** (excluding comments)

Mathlib covers a broad range of undergraduate-level mathematics and is frequently updated by the Lean community. Its structure and rigor make it an excellent dataset for language modeling tasks.

## Methodology

1. **Data Preprocessing:**
   - **Comment Removal:**  
     We strip out all Lean comment lines (`--`, `/- ... -/`) to obtain cleaner data.
   
   - **For FCN:**  
     Each line is encoded as a sequence of Unicode code points, zero-padded to length 102.
   
   - **For GPT2-lean:**  
     We insert a special `<N>` token to represent new lines and use byte-level BPE tokenization (vocab size ~23,018) including `<N>` as a special token.

2. **Models:**
   - **FCN:**  
     Inputs: Multiple zero-padded lines (Unicode-encoded).  
     Output: Next line’s Unicode-encoded representation.  
     Training: Adam optimizer, MSE loss.
   
   - **GPT2-lean:**  
     Architecture: GPT-2 with 12 layers and 12 attention heads.  
     Input: Tokenized text with `<N>` line separators.  
     Training: Language modeling objective.

3. **Training Setup:**
   - Data split: 90% training, 10% testing.
   - FCN: Various layer configurations (e.g., [2048, 512, 256]). Despite adjustments, accuracy remained low.
   - GPT2-lean: After 10 epochs and ~232.5k steps, training loss ~1.66. Exact prediction accuracy was 0.0, but outputs resembled valid Lean proof structures.

4. **Evaluation:**
   - **Quantitative:**  
     Exact match accuracy was near 0.0 for both models.
   
   - **Qualitative:**  
     FCN output was largely nonsensical. GPT2-lean output, while incorrect, was structurally coherent and often syntactically plausible.

## Repository Structure

- **`preprocess/`**: Preprocessed Lean files for training/testing.
- **`runs/`**: Saved checkpoints and configurations.

## Results

- **FCN:**  
  Best training loss ~111.9 MSE, but test accuracy 0.0. Predictions didn’t resemble Lean code.

- **GPT2-lean:**  
  Though accuracy also measured 0.0, predictions were syntactically structured and semantically closer to valid Lean code, showing a learned pattern recognition.

## Conclusions and Future Work

- **Conclusions:**  
  FCN struggled due to naive encoding of tokens. GPT2-lean showed promise, producing more meaningful lines even if exact correctness was absent.

- **Future Work:**  
  - Experiment with other language models like BERT.
  - Expand to other proof assistants.
  - Explore improved tokenization and embedding techniques.

## Acknowledgments

- **Lean Community & Mathlib Contributors:** For providing a rich dataset.
- **Hugging Face & Open-Source Tools:** For enabling accessible fine-tuning of language models.

---

Thank you for your interest in this project. We encourage you to explore the repository, run the code, and build upon this work.

