"""Vercel serverless function: used-car value prediction (stdlib only).

Reproduces the repo's sklearn pipeline without sklearn/pandas:
  median/mode imputation -> StandardScaler -> OneHotEncoder(ignore)
  -> Ridge dot product -> expm1 -> max(0, .)
Weights: api/model.json (exported from the trained pipeline).

  POST /api/predict  {name, fuel, seller_type, transmission, owner,
                      year, km_driven, mileage_value, engine_value,
                      max_power_value, seats}
  -> {"predicted_price": ...}
"""
import json
import math
import os
from http.server import BaseHTTPRequestHandler

with open(os.path.join(os.path.dirname(__file__), "model.json")) as f:
    M = json.load(f)

NUM = M["numeric_features"]
NUM_MED = M["numeric_median"]
MEAN = M["scaler_mean"]
SCALE = M["scaler_scale"]
CATS = M["categorical_features"]
CAT_MODE = M["categorical_mode"]
CATEGORIES = M["categories"]
COEF = M["coef"]
INTERCEPT = M["intercept"]


def _num(value, median):
    try:
        v = float(value)
        if v != v:  # NaN
            return median
        return v
    except (TypeError, ValueError):
        return median


def predict(car):
    total = INTERCEPT
    j = 0
    for i, feat in enumerate(NUM):
        v = _num(car.get(feat), NUM_MED[i])
        total += ((v - MEAN[i]) / SCALE[i]) * COEF[j]
        j += 1
    for k, feat in enumerate(CATS):
        val = car.get(feat) or CAT_MODE[k]
        for cat in CATEGORIES[feat]:
            if val == cat:
                total += COEF[j]
            j += 1
    return max(0.0, math.expm1(total))


class handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            car = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            self._json({"error": "invalid JSON body"}, status=400)
            return
        if not isinstance(car, dict) or not car.get("name"):
            self._json({"error": "missing 'name' (vehicle variant)"}, status=400)
            return
        try:
            price = predict(car)
        except (ValueError, TypeError, KeyError) as exc:
            self._json({"error": f"bad input: {exc}"}, status=400)
            return
        self._json({"predicted_price": round(price, 2)})
