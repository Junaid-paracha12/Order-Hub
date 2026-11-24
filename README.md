# E-Commerce Order Management Hub

Orders come in → a **delivery promise** and late-risk is computed per order → the **warehouse (WMS)** gets next-week demand and reorder levels per category → the **customer portal** shows the order with its promise and tracking events.
Models are trained on the public Olist Brazilian e-commerce data (`notebooks/Olist_Complete_Project.ipynb`).

```
olist-order-hub/
├── app/
│   ├── main.py           FastAPI service + serves the frontend
│   └── pipeline.py       feature building + inference (delivery, demand, stock actions, portal view)
├── frontend/index.html   one-file UI with two screens: 1 Delivery promise (model 1) · 2 Warehouse demand (model 2)
├── models/               delivery_model.joblib, demand_model.joblib   (output of the notebook)
├── data/                 reference.json (route / state / seller history), weekly_units.csv, sample_orders.json, report.json
├── scripts/build_reference.py   rebuilds data/* from the notebook's olist_orders_clean.csv + olist_combined.csv
├── tests/test_hub.py     12 tests (API, sanity of predictions, no look-ahead in the demand features, notebook parity)
├── notebooks/            the complete notebook (combine 9 tables → clean/impute → EDA → regression → demand forecast)
├── Dockerfile · requirements.txt
```

## Run
```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000        # open http://localhost:8000   (API docs: /docs)
python -m pytest -q tests                          # 12 tests
```
or `docker build -t order-hub . && docker run -p 8000:8000 order-hub`.
The pinned library versions matter (scikit-learn pickles). The XGBoost booster is stored as JSON inside the joblib (`scripts/make_portable.py`), so it loads on Windows/Linux and across XGBoost builds. `xgboost-cpu` provides the same `xgboost` Python module with CPU-only inference and avoids the larger GPU dependencies in Vercel's function bundle.

## Deploy to Vercel
Import the repository with its root directory set to the repository root. Vercel detects the FastAPI app in `app/main.py`; `vercel.json` includes the data, frontend, and model files that the function reads at runtime. After deployment, check the Vercel Runtime Logs if `/api/health` still fails.

## Customer-chosen delivery date
On screen 1 the customer's wished date can be entered (and freight left empty). The hub returns the typical **freight** for the route and weight band, the **chance** that the parcel arrives by the wished date, and the earliest dates it can promise at 80 / 90 / 94 / 99 % certainty. Freight cannot buy speed: inside one route and weight band, freight and delivery days are almost uncorrelated (≈ -0.07), and the Olist data has no express option, so the date is something the shop promises, not something priced.
The promise multiplier is **calibrated** (`scripts/calibrate.py` → `data/calibration.json`): a raw ratio quantile of 0.94 would give ≈ 99 % on-time, not 94 %. Calibration used 2,500 test-period orders replayed through this app.

## API
| Method | Path | What |
|---|---|---|
| GET | `/api/freight` | typical freight for a route and weight |
| POST | `/api/delivery/predict` | order fields (freight optional, `desired_date` optional) → expected days, promise days/date at a chosen on-time probability, late risk, alert |
| GET | `/api/orders`, `/api/orders/{id}` | scored sample shop feed; portal view with timeline |
| GET | `/api/demand?week=&service_level=` | forecast units + reorder level for the 20 largest categories |
| POST | `/api/demand/stock` | `{"stock": {"auto": 40}}` → order quantity and OK/REORDER per category |
| GET | `/api/demand/history`, `/api/meta`, `/api/report`, `/api/health` | chart data, form options, model tables |

Example: `curl -X POST localhost:8000/api/delivery/predict -H 'content-type: application/json' -d '{"seller_state":"SP","customer_state":"RJ","freight":20,"weight_g":1500,"purchase_time":"2018-08-20 14:30"}'`

## What the models can and cannot do (read this before presenting)
* **Delivery days per order: R² ≈ 0.25, MAE ≈ 3.4 days** (baseline "median" predictor: MAE 5.2). RMAE 0.66, WAPE 0.39. This is the ceiling of the data: seller hand-over delay, weather and strikes are not known at order time (an upper bound of ≈0.28 was measured; ≈0.44 if the carrier hand-over time is known). The value is in the **promise**: 18.0 days instead of 21.4 at the same on-time rate on the notebook's test set, and a late-risk score (AUC 0.81) that flags the risky orders (precision 14 %, recall 72 %).
* **Weekly category demand: R² ≈ 0.79, MAE ≈ 18.7 units, WAPE 0.26**, but the naive "same as last week" forecast already has R² 0.73, so the real gain is ≈ 8 % lower MAE. Reorder levels (q90) covered 91.6 % of the test weeks.
* Olist has **no stock-level data**, so the WMS tab compares a forecast with the stock *you type in*; it does not read a real WMS and no inventory saving is measured.
* Orders and tracking come from dataset timestamps. A real deployment would replace `sample_orders.json` with the Shopify order webhook and the carrier tracking feed, and push the promise date back to the shop / portal.
* The history features (recent speed of a route, a seller's hand-over time) are frozen at the data's end (`reference.json`, as of 2018-10-17). In production this table has to be refreshed daily from delivered orders (`scripts/build_reference.py` shows the exact windows: route 45 d, destination 21 d, seller 60 d).
* `sample_orders.json` is a 45-day window of May–June 2018 for the demo only; it is not a benchmark (the benchmark is the notebook's time-based test split).
