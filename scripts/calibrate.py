"""Calibrate the promise: which ratio-quantile gives which real on-time rate?
The stored ratio quantiles come from out-of-fold training weeks, which were slower than the test period, so "quantile 0.94" is NOT "94 % on time".
This script rebuilds the model inputs for a sample of TEST-period orders, scores them and measures the real on-time rate for every quantile level.
usage: python scripts/calibrate.py olist_orders_clean.csv          -> data/calibration.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import pipeline as P

root = P.ROOT
dl = pd.read_csv(sys.argv[1], parse_dates=['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date'])
dl = dl[dl.order_status.eq('delivered') & dl.days.notna()].reset_index(drop=True)
cut = dl.order_purchase_timestamp.quantile(.80); emb = P._delivery()['embargo_days']
test = dl[(dl.order_purchase_timestamp > cut) & (dl.order_purchase_timestamp <= dl.order_purchase_timestamp.max() - pd.Timedelta(days=60))]   # recent orders are delivered only if fast: stop 60 days before the end
s = test.sample(min(2500, len(test)), random_state=3)
g_carr = float(dl.carrier_days.median())
pred, act = [], []
for r in s.itertuples():
    t = r.order_purchase_timestamp; hist = dl[dl.order_delivered_customer_date < t]
    w14 = hist[hist.order_delivered_customer_date >= t - pd.Timedelta(days=14)].days.mean(); w14 = w14 if w14 == w14 else dl.days.mean()
    def rm(mask, val, win, k, prior, known='order_delivered_customer_date'):
        h = hist[mask(hist)]; h = h[h[known] >= t - pd.Timedelta(days=win)]; return float((h[val].sum() + k * prior) / (len(h) + k))
    inp = {'seller_id': r.seller_id, 'seller_state': r.seller_state, 'customer_state': r.customer_state, 'freight': float(r.freight), 'weight_g': float(r.weight_g), 'n_items': int(r.n_items), 'n_sellers': int(r.n_sellers),
           'payment_type': r.payment_type if isinstance(r.payment_type, str) else 'credit_card', 'est_days': float(r.est_days), 'limit_days': float(r.limit_days), 's_lat': float(r.s_lat), 'purchase_time': str(t),
           'recent_route_days': rm(lambda h: (h.seller_state == r.seller_state) & (h.customer_state == r.customer_state), 'days', 45, 5, w14),
           'recent_dest_days': rm(lambda h: h.customer_state == r.customer_state, 'days', 21, 5, w14),
           'recent_seller_carrier': rm(lambda h: h.seller_id == r.seller_id, 'carrier_days', 60, 3, g_carr, 'order_delivered_carrier_date'), 'seller_backlog': 0.0}
    pred.append(float(np.clip(P._delivery()['model'].predict(P.delivery_features(inp))[0], .5, None))); act.append(float(r.days))
pred, act = np.array(pred), np.array(act); q = P._delivery()['ratio_quantiles']; levels = np.linspace(.01, .99, 99)
on_time = np.array([np.mean(act <= pred * q[i]) for i in range(99)])
on_time = np.maximum.accumulate(on_time)
print('test orders', len(s), '| MAE', round(np.abs(act - pred).mean(), 3), '| R2', round(1 - ((act - pred) ** 2).sum() / ((act - act.mean()) ** 2).sum(), 3))
for lv in (.5, .8, .85, .9, .95, .97): print(f'ratio quantile {lv:.2f} -> on-time {on_time[int(round(lv * 100)) - 1]:.3f}')
json.dump({'quantile_level': levels.round(2).tolist(), 'on_time': on_time.round(4).tolist(), 'n_orders': int(len(s))}, open(root / 'data' / 'calibration.json', 'w'))
