# Olist E-commerce Analytics Dashboard

Interactive Streamlit dashboard and machine learning pipeline built on the Olist Brazilian e-commerce dataset.

## Project Summary
This project combines business analytics and predictive modeling to support e-commerce decision-making.

Key capabilities:
- Executive KPI analysis by customer state and product category
- Delivery time prediction
- Late-delivery risk prediction
- Customer satisfaction prediction
- Customer segmentation with KMeans

## Tech Stack
- Python
- Streamlit
- pandas, NumPy
- scikit-learn
- Plotly

## Repository Structure
- `dashboard.py`: Streamlit application
- `train_models.py`: model training and artifact export script
- `functions.py`: plotting utilities used by dashboard views
- `kaggle_dataset_prep.py`: optional Kaggle data download and preprocessing helper
- `models/`: serialized model artifacts (`.pkl`)
- `olist_cleaner_dataset.csv`: cleaned dataset used by the dashboard and training script
- `requirements.txt`: Python dependencies

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`

## Run Locally
1. Train models (if model artifacts are missing):
   - `python train_models.py`
2. Launch dashboard:
   - `streamlit run dashboard.py`

## Reproducibility Notes
- The dashboard expects model files under `models/`.
- If model files are missing, dashboard pages that require predictions will show guidance messages.
- The optional Kaggle helper script expects `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

## Data Policy
- Raw source CSV files are not committed to this repository.
- Download raw data locally using `kaggle_dataset_prep.py`.
- Keep this repository focused on code, documentation, and curated project outputs.

## Data Source & License
- Dataset: [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- Source platform: Kaggle
- Usage note: Follow the dataset license and terms on Kaggle before redistribution or commercial reuse.

Example workflow:
1. Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your environment.
2. Run `python kaggle_dataset_prep.py` to download and prepare data locally.
3. Run `python train_models.py`, then `streamlit run dashboard.py`.

## Portfolio Context
This repository is shared as a portfolio-ready version of a university team project.

I focused on the application layer and integration, including:
- Streamlit dashboard implementation and interaction design
- Model loading and prediction pipelines
- Analytics views for summary, risk optimization, and segmentation
- Deployment-oriented cleanup and repository documentation

## Team Collaboration
The original project work was completed in a 4-member academic team spanning:
- Data preprocessing and integration
- Data mining and model evaluation
- Report writing and presentation

## Privacy
To protect student privacy, this public repository excludes names, initials, and student IDs.