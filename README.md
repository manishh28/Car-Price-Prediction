# carpricedekhoo

A web app that estimates the resale value of a used car from 8,000+ listing records.
Live as a static frontend + Python serverless API on Vercel.

## Features

- Two searchable suggestion dropdowns for brand and model
- Correct handling for multi-word brands such as Land Rover and Ashok Leyland
- Typical engine specifications filled from the selected model
- Personalized estimated resale price from the trained prediction model
- Compact vehicle detail choices and optional engine specifications
- Responsive desktop and mobile interface

## Project layout

```text
web/
  index.html      carpricedekhoo estimator page
  style.css       Brand styles, local fonts, and responsive layout
  app.js          Vehicle picker and live model prediction flow
  assets/         Logo and favicon
  fonts/          Bonum and Inter font files
  data.json       Brands, models, typical specs, price stats, metrics
  api/
    predict.py    Serverless prediction endpoint (stdlib only, no ML deps)
    model.json    Exported Ridge weights + imputer/scaler stats
```

The API reproduces the trained Ridge-on-log-price pipeline by hand
(median/mode imputation → scaling → one-hot → dot product → expm1),
so the serverless function stays dependency-free and far under
Vercel's function size limit.

## Run locally

Any static server works for the frontend, e.g.:

```bash
cd web
python -m http.server 8000
```

Prediction calls (`POST /api/predict`) need `vercel dev` or the deployed site.

## Retrain / refresh data

Regenerate `web/api/model.json` + `web/data.json` from `car details.csv`
with the export script, then commit and push — Vercel redeploys automatically.
