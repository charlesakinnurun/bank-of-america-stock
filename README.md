# Bank of America Stock
![bank-of-america](/image.jpg)

## Procedures
- Import the libraries
    - pandas
    - scikit-learn
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Data Acquisition 
    - Data acquired from the Yahoo Finance API
- Data Loading
- Feature Engineering
- Data Preparation
- Data Splitting
    - Split the data into training and testing sets
    -  We use a time-series split (shuffling is set to False) to avoid future data leaking into the past
- Data Scaling
    - Initialize the StandardScaler and fit it only on the training data
- Pre-Training Visualization

![pre-training-visualization](/output1.png)
- Model Comparison
- Model Training 
- Hyperparameter Tuning
- Post-Training Visualization

![post-training-visualization](/output2.png)
- New Prediction Input Function

## Tech Stack and Tools
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Environment
    - Jupyter Notebook
    - Anaconda
    - Google Colab
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```



## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/bank-of-america-stock.git
cd bank-of-america-stock
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Project Structure
```
bank-of-america-stock/
│
├── model.ipynb  
|── model.py    
|── BAC_stock_data.csv  
├── requirements.txt 
├── image.jpg    
├── output1.png
├── output2.png          
├── CONTRIBUTING.md    
├── CODE_OF_CONDUCT.md 
├── LICENSE
└── README.md          

```