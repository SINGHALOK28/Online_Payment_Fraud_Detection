# Online Payment Fraud Detection

A machine learning project to detect fraudulent online payment transactions using various classification algorithms.

## Project Overview

git commit -m "Initial commit: Online Payment Fraud Detection project setup"

## Dataset

The project uses the `onlinefraud.csv` dataset containing payment transaction records with features for fraud detection.

**To get the dataset, use one of these methods:**

### Method 1: Download from Kaggle (Recommended)
```bash
# Install Kaggle CLI
pip install kaggle

# Download the dataset
kaggle datasets download -d ealaxy/paysim1

# Extract and rename
unzip paysim1.zip
mv PS_20174392719_1491204493457_log.csv onlinefraud.csv
```

### Method 2: Manual Download
- Visit: https://www.kaggle.com/datasets/ealaxy/paysim1
- Click "Download" button
- Extract the ZIP file
- Rename the CSV to `onlinefraud.csv`
- Place it in the project root folder

### Method 3: Use a Sample Dataset
If the full dataset is too large, create a sample CSV with the same structure (see onlinefraud_sample.csv in the repo)

## Project Structure

```
Online_Payment_Fraud_Detection/
├── Images/
│   └── visualise.png
├── main.ipynb
├── onlinefraud.csv
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
├── LICENSE
└── CODE_OF_CONDUCT.md
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

## Features

- Data preprocessing and exploration
- Feature engineering
- Model training and evaluation
- Cross-validation
- Model comparison
- Visualization of results

