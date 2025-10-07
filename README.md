

```markdown
# 🧠 AlpaCare — Medical Instruction Assistant

📦 **GitHub Repository:** [https://github.com/atharv-17-L/AlpaCare](https://github.com/atharv-17-L/AlpaCare)

---

## 📘 Overview

**AlpaCare** is a fine-tuned **DistilGPT-2** language model built to provide **safe and educational medical guidance**.  
It uses **LoRA (Low-Rank Adaptation)** via **PEFT** for lightweight and efficient fine-tuning on the **AlpaCare-MedInstruct-52k** dataset from Hugging Face.  

The project demonstrates how small transformer models can be adapted for **domain-specific, safe response generation** in healthcare contexts.

> ⚠️ **Disclaimer:** This model is for educational purposes only and does not provide medical advice.

---

## 🧩 Key Features

- 🧠 Fine-tunes **DistilGPT-2** using **LoRA (PEFT)**  
- ⚙️ Implemented completely on **Google Colab**  
- 🗂️ Dataset: `lavita/AlpaCare-MedInstruct-52k`  
- 💾 Produces a reusable **LoRA adapter**  
- ⚡ Lightweight, low-cost fine-tuning  
- 🔐 Includes disclaimers for medical safety  

---

## 📁 Repository Structure

```

AlpaCare/
│
├── data_loader.py                 # Loads and preprocesses dataset
│
├── notebooks/
│   ├── colab_finetune.ipynb       # Fine-tuning notebook
│   └── inference_demo.ipynb       # Inference and testing notebook
│
├── dataset/
│   └── train-00000.parquet        # Source dataset file (sample)
│
├── lora_adapter.zip               # Trained LoRA adapter (model weights)
│
├── LICENSE                        # MIT License
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies list

````

---

## ⚙️ Setup and Execution Guide

You can run this project on **Google Colab** (recommended) or locally on your computer.

---

### 🧭 Option 1 — Run on Google Colab (Recommended)

1. Open the notebook:  
   👉 [`notebooks/colab_finetune.ipynb`](./notebooks/colab_finetune.ipynb)
2. Run all cells step-by-step:
   - Install dependencies  
   - Load dataset (`data_loader.py`)  
   - Load tokenizer and model (`DistilGPT-2`)  
   - Configure **LoRA (r=8, α=32, dropout=0.1)**  
   - Train and save adapter  
3. After training, download the generated folder **`lora_adapter`** for inference.

---

### 🧭 Option 2 — Run Locally (on Your PC)

#### 1️⃣ Clone the Repository
```bash
git clone https://github.com/atharv-17-L/AlpaCare.git
cd AlpaCare
````

#### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Launch the Notebook

Open and run:

```
notebooks/colab_finetune.ipynb
```

---

## 🧮 How to Run Inference

After fine-tuning, open **`inference_demo.ipynb`** and run:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, "./lora_adapter")

def generate_response(prompt):
    input_text = f"Instruction: {prompt}\n\nResponse:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text + "\n\n⚠️ Disclaimer: Educational only. Consult a clinician."

print(generate_response("What are healthy bedtime habits?"))
```

---

## 📊 Sample Input / Output
AlpaCare/
│
├── data_loader.py # Loads and preprocesses the dataset
│
├── notebooks/
│ ├── colab_finetune.ipynb # Fine-tuning notebook
│ └── inference_demo.ipynb # Inference and testing notebook
│
├── dataset/
│ └── train-00000.parquet # Sample dataset file
│
├── lora_adapter.zip # Trained LoRA adapter (model weights)
│
├── LICENSE # MIT License
├── README.md # Project documentation
└── requirements.txt # Python dependencies list

---

## 🧠 LoRA Configuration Details

| Parameter |   Value   | Description          |
| :-------- | :-------: | :------------------- |
| Rank (r)  |     8     | Adapter layer rank   |
| Alpha (α) |     32    | Scaling factor       |
| Dropout   |    0.1    | Prevents overfitting |
| Task Type | Causal LM | Text generation task |

💡 These hyperparameters ensure efficient and memory-friendly fine-tuning suitable for Google Colab GPUs.

---

## 📈 Evaluation

* ✅ Human evaluation: 30 outputs tested
* ✅ Metrics: *Clarity*, *Safety*, *Relevance*, *Fluency*
* ⚙️ Optional automatic metrics: BLEU, Rouge
* 🧾 Results summarized in project report

---

## 🧾 Dependencies

All dependencies are in `requirements.txt`.
Or install manually:

```bash
pip install transformers datasets peft loralib torch pandas pyarrow accelerate scikit-learn
```

---

## 🛡️ Ethical & Safety Statement

* The model **does not provide medical advice**.
* Every output includes a **medical disclaimer**.
* Built using ethical AI guidelines from Hugging Face.

---

## 📘 References

* Dataset: [lavita/AlpaCare-MedInstruct-52k](https://huggingface.co/datasets/lavita/AlpaCare-MedInstruct-52k)
* Model: [DistilGPT-2](https://huggingface.co/distilgpt2)
* Frameworks: Hugging Face Transformers, PEFT, Datasets, Accelerate

---

## 📄 License

This project is released under the **Apache 2.0** — free for educational and research use only.

---

## 👤 Author

**Atharv Latta**
Solar Industries India Ltd — AIML Internship Assessment (2025)

---

## ✅ Submission Checklist

| Item                  |   Status  |
| --------------------- | :-------: |
| Public GitHub Repo    |     ✅     |
| README.md (Detailed)  |     ✅     |
| LoRA Adapter Uploaded |     ✅     |
| Fine-tuning Notebook  |     ✅     |
| Inference Notebook    |     ✅     |
| Dataset (≤100 MB)     |     ✅     |
| License               |     ✅     |
| JotForm Submission    | ⏳ Pending |

```

---


---

Would you like me to now generate a **matching `requirements.txt`** file (so you can upload that too right away)?
```
