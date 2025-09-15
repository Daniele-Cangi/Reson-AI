<img width="1536" height="1024" alt="Reson" src="https://github.com/user-attachments/assets/ab5dc731-2ab3-4c10-a522-b090f81cffd0" />

# Reson — LLaMA-2 7B HF Fine-Tuned
[![CI](https://github.com/Daniele-Cangi/Reson-AI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Daniele-Cangi/Reson-AI/actions/workflows/ci.yml)

Reson is a **fine-tuned version of LLaMA-2 7B HF**, trained on a custom dataset of ~11,000 examples.  
It is designed as a **decision-support and cognitive simulation system**, not as a generic chatbot.  
The goal is to explore **extreme meta-cognition, multi-domain reasoning, and strategic adaptability**.

---
🔗 Model Links

🤗 Hugging Face Model: [Nexus-Walker/Reson](https://huggingface.co/Nexus-Walker/Reson)

📊 Model Card: View on Hugging Face Hub

💾 Download: Available for download via transformers library or direct download

👁️ Demo transcripts: https://huggingface.co/Nexus-Walker/Reson/blob/main/demo_chat.md

## 🎯 Core Focus

Reson is tuned for:
- **Multi-domain reasoning** → trading, physics, biology, philosophy, strategy, and beyond.  
- **Extreme meta-cognition** → reflection, self-analysis, recursion, ambiguity handling, error monitoring.  
- **Strategic adaptability** → multi-scenario planning, counterfactuals, causal/temporal reasoning.  
- **Unconventional logic blending** → paraconsistent logic, analogy generation, reasoning without patterns.  
- **Exploratory cognition** → producing answers that may look “hallucinatory”, but are actually simulations of adaptive strategies under uncertainty.

---

## 🗂 Dataset

- **Size**: ~11,000 instruction–response pairs.  
- **Format**: JSONL (`{"instruction": "...", "response": "..."}`).  
- **Content blocks**:
  - Causal reasoning and temporal logic  
  - Recursive and paraconsistent logics  
  - Ambiguity management and second-order dissonances  
  - Cross-domain analogies and meta-analogical reasoning  
  - Strategic planning and intentional causal inversion  
  - Simulation of other intelligences and pre-memetic dynamics  

This composition pushes the model beyond factual Q&A into **adaptive, strategy-driven cognition**.

---

## ⚙️ Training Method

- **Base model**: `LLaMA-2-7B HF`  
- **Technique**: LoRA fine-tuning with 4-bit quantization (nf4)  
- **Trainable parameters**: LoRA adapters only (backbone frozen)  
- **Dataset size**: ~11k Q&A pairs focused on cognition and strategy  

This setup ensures **efficiency** (training on consumer GPUs) while embedding **new reasoning behaviors** into the model.  
The result is a system that simulates **adaptable, strategic thought** across multiple domains.

---

## 🚀 Capabilities

- **Adaptive outputs**: style shifts based on context/domain.  
- **Self-reflection**: explicitly evaluates uncertainty and suggests corrections.  
- **Strategic synthesis**: generates reasoning paths, trade-offs, and contingency plans.  
- **Cross-domain blending**: applies concepts from one field to another in creative ways.  
- **Controlled divergence**: produces exploratory answers that simulate adaptation, not static recall.

---

## 📊 Example Use Cases

- **Trading decision support** → not only signals, but rationales and scenario trees.  
- **Research companion** → generate hypotheses across science and philosophy.  
- **Education** → explain paradoxes and complex logic from multiple perspectives.  
- **Cognitive sandbox** → study emergent properties of meta-trained LLMs.

---

## ⚠️ Notes & Caveats

- Reson is **not a factual Q&A system**.  
- Outputs are exploratory and adaptive — designed for **thinking under uncertainty**.  
- Professional or commercial use requires explicit licensing.  
- Base model license: [LLaMA 2 Terms of Use](https://ai.meta.com/llama/license/).

---
