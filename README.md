# MedMCQA LLM Fine-Tuning

## Overview
This repository contains code for fine-tuning a large language model (LLM) on the **MedMCQA** dataset. MedMCQA is a large-scale, multiple-choice question answering (MCQA) dataset designed to address real-world medical entrance exam questions, including AIIMS & NEET PG entrance exams. The dataset covers **21 medical subjects** and **2,400+ healthcare topics**, making it a valuable resource for training medical QA systems.

## Model Information
The fine-tuned model is based on **Llama 3.1-8B** and utilizes **qLORA** and **PEFT** (Parameter-Efficient Fine-Tuning) for optimization. The model was trained using **Accelerate** and **DeepSpeed** on **4 H100 GPUs**.

## Dataset
**MedMCQA** consists of:
- **182,822** training questions
- **6,150** test questions
- **4,183** validation questions

Each question includes:
- A **question text**
- Four answer options (**opa, opb, opc, opd**)
- The **correct answer**
- An **expert explanation**
- The **subject and topic** of the question
- A **unique identifier**
- The **choice type** (single or multiple correct answers)

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/jindalankush28/medical-mcq-LLM-finetuning.git
cd medical-mcq-LLM-finetuning
```

### 2. Create a Virtual Environment
```bash
conda create -n medmcqa python=3.10 -y
conda activate medmcqa
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Alternatively, you can set up the environment using `environment.yml`:
```bash
conda env create -f environment.yml
conda activate medmcqa
```

### 4. Fine-Tuning the Model
To fine-tune the model using **Accelerate** and **DeepSpeed**, run:
```bash
accelerate launch sft.py
```

## Project Structure
```
├── data/                      # MedMCQA dataset
├── inference/                 # Inference results
├── medical_qa_model/          # Model directory
├── sft.py                     # Fine-tuning script
├── utils.py                   # Utility functions
├── default_config.yaml        # Configuration file
├── EDA.ipynb                  # Exploratory Data Analysis
├── environment.yml            # Conda environment configuration
├── log_deepspeed_v3.txt       # DeepSpeed logs
├── README.md                  # Project documentation
```

## Inference
Once the model is trained, you can run inference using:
```bash
python batched_inference.py
```

## Citation
If you use this dataset or model in your research, please cite the original MedMCQA paper:
```bibtex
@article{MedMCQA,
  title={MedMCQA: A Large-Scale Medical Multiple-Choice Question Answering Dataset},
  author={Kushal Agarwal et al.},
  journal={arXiv preprint arXiv:2203.09714},
  year={2022}
}
```

## License
This project is open-source and available under the **Apache 2.0 License**.

---
For any questions or contributions, feel free to open an **issue** or submit a **pull request**. 🚀

