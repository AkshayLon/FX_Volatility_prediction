# FX Volatility prediction using Gaussian Process Regression

This project aims to model and thus forecast volatility in the EUR/USD exchange rate using Gaussian Process Regression.
Inspired by <a url="https://www.oxford-man.ox.ac.uk/wp-content/uploads/2020/06/An-Overview-of-Gaussian-process-Regression-for.pdf">this paper</a>.

## Installation and Usage

Firstly (provided that pip is installed), run the following command to install the required packages:
```bash
pip install requirements.txt
```
For the example given in the code, we have used EUR/USD 30-min timeframe data from 2017 onwards. If you would like to download alternative data, it is available at <a url="https://forexsb.com/historical-forex-data">this site</a>.

An example for a forecast is given in main.py.