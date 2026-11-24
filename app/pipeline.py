"""Inference pipeline of the Order Management Hub.

Step 1  orders in      -> delivery promise + late-risk per order      (delivery_model.joblib)
Step 2  WMS            -> next-week demand + reorder level / category (demand_model.joblib)
Step 3  customer portal-> tracking view of an order with its promise

The history features (recent route / destination / seller delivery times, weekly unit counts) are looked up
from `data/reference.json` and `data/weekly_units.csv`, built by scripts/build_reference.py from the cleaned data.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BLACK_FRIDAY = pd.Timestamp('2017-11-24')


class _PortableDelivery:
    """preprocessing + XGBoost booster rebuilt from JSON; predicts days (the model was trained on log1p(days))."""
    def __init__(self, prep, booster_json):
        from xgboost import XGBRegressor
        self.prep, self.xgb = prep, XGBRegressor(); self.xgb.load_model(bytearray(booster_json.encode('utf-8')))
    def predict(self, X): return np.expm1(self.xgb.predict(self.prep.transform(X)))


@lru_cache(maxsize=1)
def _delivery() -> dict:
    d = joblib.load(ROOT / 'models' / 'delivery_model.joblib')
    if d.get('model') is None and d.get('xgb_json'): d['model'] = _PortableDelivery(d['prep'], d['xgb_json'])
    return d
@lru_cache(maxsize=1)
def _demand(): return joblib.load(ROOT / 'models' / 'demand_model.joblib')
@lru_cache(maxsize=1)
def reference() -> dict: return json.loads((ROOT / 'data' / 'reference.json').read_text())
@lru_cache(maxsize=1)
def weekly_units() -> pd.DataFrame:
    return pd.read_csv(ROOT / 'data' / 'weekly_units.csv', index_col=0, parse_dates=True)
@lru_cache(maxsize=1)
def calibration() -> dict: return json.loads((ROOT / 'data' / 'calibration.json').read_text())
@lru_cache(maxsize=1)
def sample_orders() -> list[dict]: return json.loads((ROOT / 'data' / 'sample_orders.json').read_text())


# --------------------------------------------------------------------------- delivery
def estimate_freight(seller_state: str, customer_state: str, weight_g: float) -> float:
    """Typical freight charge for the route and weight band (route if it has enough orders, else destination state, else global)."""
    f = reference()['freight']; band = str(int(np.searchsorted(f['edges'][1:-1], weight_g, side='right')))
    for tab in (f['route'].get(f'{seller_state}|{customer_state}', {}), f['dest'].get(customer_state, {}), f['global']):
        if band in tab: return float(tab[band])
    return float(f['global'][band])


def _level(ratio: float) -> float:
    """Quantile level of an actual/predicted ratio in the stored out-of-fold ratio distribution."""
    c = calibration(); return float(np.interp(ratio, _delivery()['ratio_quantiles'], c['quantile_level'], left=0.0, right=1.0))


def ratio_for_on_time(target: float) -> float:
    """Multiplier of the predicted days that gives `target` REAL on-time rate (calibrated by scripts/calibrate.py; quantile 0.94 would not mean 94 % on time)."""
    c = calibration(); lvl = float(np.interp(target, c['on_time'], c['quantile_level'])); return float(np.interp(lvl, c['quantile_level'], _delivery()['ratio_quantiles']))


def on_time_for_ratio(ratio: float) -> float:
    """Real chance that actual/predicted <= ratio."""
    c = calibration(); lvl = _level(ratio)
    return float(np.interp(lvl, [0.0] + c['quantile_level'], [0.0] + c['on_time']))


def desired_date_check(pred: float, purchase_time: pd.Timestamp, desired_date: str) -> dict:
    """Can the order arrive by the date the customer wants?  chance = P(delivery days <= days available)."""
    avail = (pd.Timestamp(desired_date) - purchase_time.normalize()).days
    levels = {f'{int(l * 100)}%': (purchase_time + pd.Timedelta(days=pred * ratio_for_on_time(l))).strftime('%Y-%m-%d') for l in (.8, .9, .94, .99)}
    if avail <= 0: return {'days_available': int(avail), 'chance': 0.0, 'verdict': 'impossible', 'earliest_dates': levels}
    chance = on_time_for_ratio(avail / pred)
    return {'days_available': int(avail), 'chance': round(chance, 3), 'verdict': 'safe' if chance >= .9 else 'risky' if chance >= .5 else 'unrealistic', 'earliest_dates': levels}


def delivery_features(o: dict) -> pd.DataFrame:
    """Raw order -> the 15 model features. Anything the caller does not know is filled from the reference tables."""
    ref, ts = reference(), pd.Timestamp(o.get('purchase_time') or pd.Timestamp.now().floor('min'))
    s_state, c_state, sid = o['seller_state'], o['customer_state'], o.get('seller_id')
    seller = ref['sellers'].get(sid or '', {})
    route = ref['routes'].get(f'{s_state}|{c_state}', {})
    f = {
        'recent_route_days': o.get('recent_route_days', route.get('recent_days', ref['global']['recent_days'])),
        'olist_est_days': o.get('est_days', route.get('est_days', ref['global']['est_days'])),
        's_lat': o.get('s_lat', seller.get('lat', ref['states'][s_state]['lat'])),
        'recent_seller_carrier': o.get('recent_seller_carrier', seller.get('carrier_days', ref['states'][s_state]['carrier_days'])),
        'limit_days': o.get('limit_days', ref['global']['limit_days']),
        'same_state': int(s_state == c_state),
        'freight': float(o['freight']) if o.get('freight') is not None else estimate_freight(s_state, c_state, float(o['weight_g'])),
        'recent_dest_days': o.get('recent_dest_days', ref['states'][c_state]['recent_dest_days']),
        'dow': ts.dayofweek, 'hour': ts.hour,
        'n_sellers': int(o.get('n_sellers', 1)), 'n_items': int(o.get('n_items', 1)),
        'seller_backlog': o.get('seller_backlog', seller.get('backlog', ref['global']['backlog'])),
        'weight_g': float(o['weight_g']),
        'payment_type': o.get('payment_type', 'credit_card'),
    }
    return pd.DataFrame([f])[_delivery()['features']]


def predict_delivery(o: dict, on_time_target: float = 0.94) -> dict:
    """Expected days, a promise that holds with `on_time_target` probability, and the risk of beating Olist's own date."""
    d = _delivery(); X = delivery_features(o)
    pred = float(np.clip(d['model'].predict(X)[0], 0.5, None))
    q = d['ratio_quantiles']                                  # 99 quantiles (1..99 %) of actual / predicted days
    promise = pred * ratio_for_on_time(on_time_target)
    est = float(X.olist_est_days.iloc[0])
    risk = float(1 - np.searchsorted(q, est / pred) / len(q))  # P(actual > the marketplace's estimate)
    out = {'expected_days': round(pred, 1), 'promise_days': round(promise, 1), 'on_time_target': on_time_target,
           'marketplace_estimate_days': round(est, 1), 'late_risk': round(risk, 3),
           'alert': bool(risk >= d['alert_threshold']), 'alert_threshold': round(d['alert_threshold'], 3),
           'features_used': {k: (round(float(v), 2) if not isinstance(v, str) else v) for k, v in X.iloc[0].items()}}
    out['freight_used'] = round(float(X.freight.iloc[0]), 2); out['freight_estimated'] = o.get('freight') is None
    if o.get('desired_date'): out['desired'] = {'date': o['desired_date'], **desired_date_check(pred, pd.Timestamp(o.get('purchase_time') or pd.Timestamp.now().floor('min')), o['desired_date'])}
    if o.get('purchase_time'):
        out['promise_date'] = (pd.Timestamp(o['purchase_time']) + pd.Timedelta(days=promise)).strftime('%Y-%m-%d')
    return out


# --------------------------------------------------------------------------- demand
CATS_SORTED = lambda: sorted(_demand()['categories'])


def demand_features(week: pd.Timestamp) -> pd.DataFrame:
    """Features for the forecast of `week` (a Monday) using only weeks strictly before it."""
    u = weekly_units(); hist = u[u.index < week]
    if len(hist) < 9: raise ValueError('need at least 9 complete weeks before the forecast week')
    tot = hist.sum(axis=1); rows = []
    bf_w = int(week <= BLACK_FRIDAY <= week + pd.Timedelta(days=6)); bf_a = int(BLACK_FRIDAY < week <= BLACK_FRIDAY + pd.Timedelta(days=13))
    for cat in CATS_SORTED():
        s = hist[cat]; last4, last8 = s.iloc[-4:], s.iloc[-8:]
        rows.append({'cat': cat, 'lag1': s.iloc[-1], 'lag2': s.iloc[-2], 'lag3': s.iloc[-3], 'lag4': s.iloc[-4], 'ma4': last4.mean(), 'ma8': last8.mean(),
                     'std4': last4.std(), 'trend': s.iloc[-1] / (last4.mean() + 1), 'mkt_lag1': tot.iloc[-1], 'mkt_ma4': tot.iloc[-4:].mean(),
                     'share_lag1': s.iloc[-1] / (tot.iloc[-1] + 1), 'woy': int(week.isocalendar().week), 'bf_week': bf_w, 'bf_after': bf_a,
                     'cat_c': CATS_SORTED().index(cat)})
    return pd.DataFrame(rows)


def forecast_demand(week: str | None = None, service_level: float = .90) -> list[dict]:
    """Expected units per category for `week` (default: the week after the last complete one) and a reorder level."""
    u = weekly_units(); wk = pd.Timestamp(week) if week else u.index.max() + pd.Timedelta(days=7)
    wk = wk - pd.Timedelta(days=wk.dayofweek)
    m = _demand(); X = demand_features(wk)
    z = m['model'].predict(X[m['features']])
    if m['target_form'] == 'log1p': pred = np.expm1(z)
    elif m['target_form'] == 'growth-normalised': pred = np.expm1(z + np.log1p(X.ma4.to_numpy()))
    else: pred = np.clip(z, 0, None)
    lvl = np.ceil(np.clip(pred, 1, None) * float(np.interp(service_level, np.linspace(.01, .99, 99), m['ratio_quantiles'])))
    return [{'category': c, 'week': wk.strftime('%Y-%m-%d'), 'last_week_units': int(l), 'ma4': round(float(a), 1), 'forecast_units': round(float(p), 1),
             'reorder_level': int(r), 'service_level': service_level}
            for c, l, a, p, r in zip(X.cat, X.lag1, X.ma4, pred, lvl)]


def stock_actions(forecast: list[dict], stock: dict[str, float]) -> list[dict]:
    """Compare current WMS stock with the reorder level: how many units to order up to the level."""
    out = []
    for f in forecast:
        have = float(stock.get(f['category'], 0)); need = max(0, f['reorder_level'] - have)
        out.append({**f, 'stock': have, 'order_qty': int(np.ceil(need)), 'status': 'REORDER' if need > 0 else 'OK'})
    return out


# --------------------------------------------------------------------------- portal
def portal_view(order_id: str) -> dict | None:
    for o in sample_orders():
        if o['order_id'] == order_id:
            return {**o, 'prediction': predict_delivery({**o['input'], 'purchase_time': o['purchase_time']})}
    return None
