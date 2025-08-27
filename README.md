"""# Inventory Demand Forecasting

This project predicts item-level daily sales across multiple stores using Machine Learning.  
We compare **Random Forest, XGBoost, and LightGBM** on a dataset of 900k+ sales records.

## Features

- Preprocessing of large-scale sales data
- Feature engineering: lagged sales, day-of-week, month, store & item encoding
- Model comparison: Random Forest, XGBoost, LightGBM
- Evaluation with RMSE

## Repository Structure

- data/ → dataset
- notebooks/ → Jupyter notebook with experiments
- src/ → Python scripts for reproducibility
- requirements.txt → dependencies

## Results

- LightGBM achieved the best performance with lowest RMSE.
- Models improved forecast accuracy compared to naive approaches.

## How to Run

```bash
# Clone repo
git clone https://github.com/yourusername/demand-forecasting-ml.git
cd demand-forecasting-ml

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook notebooks/forecasting.ipynb
"""