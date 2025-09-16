# CheMLT-F
A Framework for standardized drug processing along with a family of pre-trained Multitask Transformer Models (DeBERTa) adapted for Chemical Analysis, Drug Property and Binding Affinity Prediction.

The models are fully extendable to additional datasets and are simple to modify & retrain under a standardized pipeline.

At the moment, the models support 13 different drug datasets

<img width="549" height="340" alt="image" src="https://github.com/user-attachments/assets/80a25315-04a0-475b-a064-8da07dc7a05f" />

## 🛠️ Setup

This project provides **two ways** to install dependencies:

### Option 1: Conda (recommended)

Create a new environment from the provided YAML file:

```bash
conda env create -f environment.yml
conda activate myenv
```

### Option 2: Pip (alternative)

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

- **On Linux / macOS**
```bash
source venv/bin/activate
```

- **On Windows (PowerShell)**
```powershell
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```



Due to Github Storage limitations, single-task versions of the models and original pre-training datasets are also available on Google Drive via link below
https://drive.google.com/drive/folders/1SWQ5K7EoaCpyg5s7yqRQHveTy4unsusr?usp=sharing


For additional questions, please contact the authors
