"""Make the delivery model independent of XGBoost's binary pickle format.
A pickled XGBRegressor can fail to load on another OS / XGBoost build ("input stream corrupted").
This stores the booster as JSON text inside the joblib (keys: prep, xgb_json) and app/pipeline.py rebuilds the predictor on load.
usage: python scripts/make_portable.py models/delivery_model.joblib   (run it where the model still loads, e.g. Kaggle)
"""
import sys, joblib
p = sys.argv[1]; d = joblib.load(p); m = d.get('model')
if m is None: print('already portable'); sys.exit()
last = m.regressor_.steps[-1][1]
if type(last).__name__ != 'XGBRegressor': print('delivery model is not XGBoost - nothing to do'); sys.exit()
d['prep'] = m.regressor_.steps[0][1]; d['xgb_json'] = bytes(last.get_booster().save_raw('json')).decode('utf-8'); d['model'] = None
joblib.dump(d, p); print('saved', p)
