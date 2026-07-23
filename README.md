# Used Car Value Estimator

A Streamlit app that estimates the resale value of a used car from 8,000+ listing records.

## Features

- Two-field brand and model search with relevance-ranked matching
- Correct handling for multi-word brands such as Land Rover and Ashok Leyland
- Typical specifications filled from comparable listings
- Cached preprocessing and model training
- Estimated price, comparable median, and market range
- Responsive desktop and mobile interface

## Run locally

```bash
git clone https://github.com/manishh28/Car-Price-Prediction.git
cd Car-Price-Prediction
pip install -r requirements.txt
streamlit run main.py
```
