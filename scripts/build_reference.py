"""Build data/reference.json, data/weekly_units.csv and data/sample_orders.json from the notebook outputs.
usage: python scripts/build_reference.py olist_orders_clean.csv olist_combined.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

root = Path(__file__).resolve().parent.parent; (root / 'data').mkdir(exist_ok=True)
clean = pd.read_csv(sys.argv[1], parse_dates=['order_purchase_timestamp', 'order_delivered_carrier_date', 'order_delivered_customer_date'])
comb = pd.read_csv(sys.argv[2], usecols=['order_id', 'order_item_id', 'order_purchase_timestamp', 'product_category_name_english'], parse_dates=['order_purchase_timestamp'])
dl = clean[clean.order_status.eq('delivered') & clean.days.notna()].copy()
ASOF = dl.order_purchase_timestamp.max()      # not the last delivery: after the last purchase only stragglers (very late parcels) are still being delivered, which the model never saw as 'recent'


def recent(frame, key_cols, value, known_at, asof, window, k, prior):
    """Shrunk mean of `value` of rows whose outcome was known in the `window` days before `asof`."""
    m = frame[(frame[known_at] < asof) & (frame[known_at] >= asof - pd.Timedelta(days=window))]
    g = m.groupby(key_cols)[value].agg(['sum', 'count'])
    return ((g['sum'] + k * prior) / (g['count'] + k)).to_dict()


g_recent = dl[(dl.order_delivered_customer_date >= ASOF - pd.Timedelta(days=14))].days.mean()
gl = {'recent_days': float(g_recent), 'est_days': float(dl.est_days.median()), 'limit_days': float(dl.limit_days.median()),
      'backlog': 0.0, 'carrier_days': float(dl.carrier_days.median())}

# state level
st = {}
rd = recent(dl, 'customer_state', 'days', 'order_delivered_customer_date', ASOF, 21, 5, g_recent)
for s, grp in pd.concat([dl[['seller_state', 's_lat', 'carrier_days']].rename(columns={'seller_state': 'st'}),
                         dl[['customer_state']].rename(columns={'customer_state': 'st'})]).groupby('st'):
    pass
seller_states = dl.groupby('seller_state').agg(lat=('s_lat', 'median'), carrier_days=('carrier_days', 'median'))
cust_states = sorted(dl.customer_state.unique())
for s in sorted(set(cust_states) | set(seller_states.index)):
    st[s] = {'lat': float(seller_states.lat.get(s, dl.s_lat.median())), 'carrier_days': float(seller_states.carrier_days.get(s, gl['carrier_days'])),
             'recent_dest_days': float(rd.get(s, g_recent))}
# routes
rt = recent(dl, ['seller_state', 'customer_state'], 'days', 'order_delivered_customer_date', ASOF, 45, 5, g_recent)
est = dl.groupby(['seller_state', 'customer_state']).est_days.median()
routes = {f'{a}|{b}': {'recent_days': float(rt.get((a, b), g_recent)), 'est_days': float(est[(a, b)])} for (a, b) in est.index}
# sellers
cd = dl[dl.order_delivered_carrier_date.notna()]
sc = recent(cd, 'seller_id', 'carrier_days', 'order_delivered_carrier_date', ASOF, 60, 3, gl['carrier_days'])
top = dl.seller_id.value_counts().head(600).index
slat = dl.groupby('seller_id').agg(lat=('s_lat', 'median'), state=('seller_state', 'first'))
sellers = {s: {'lat': float(slat.lat[s]), 'state': slat.state[s], 'carrier_days': float(sc.get(s, gl['carrier_days'])), 'backlog': 0.0} for s in top}
# typical freight charge by weight band: per route when it has >= 15 orders in the band, else per destination state, else global
EDGES = [0, 500, 1000, 2000, 5000, 10000, 1e9]; fb = dl.assign(band=pd.cut(dl.weight_g, EDGES, labels=False))
def med(g): return {int(b): float(v) for b, v in g.groupby('band').freight.median().items()}
rg = fb.groupby(['seller_state', 'customer_state', 'band']).freight.agg(['median', 'count']).reset_index(); rg = rg[rg['count'] >= 15].rename(columns={'median': 'med'})
freight = {'edges': EDGES[:-1] + [1e9], 'global': med(fb), 'dest': {s_: med(g) for s_, g in fb.groupby('customer_state')},
           'route': {f'{a}|{b}': {int(r.band): float(r.med) for r in g.itertuples()} for (a, b), g in rg.groupby(['seller_state', 'customer_state'])}}
ref = {'asof': str(ASOF.date()), 'freight': freight, 'global': gl, 'states': st, 'routes': routes, 'sellers': sellers,
       'payment_types': sorted(dl.payment_type.dropna().unique().tolist()),
       'customer_states': cust_states, 'seller_states': sorted(seller_states.index.tolist())}
(root / 'data' / 'reference.json').write_text(json.dumps(ref, separators=(',', ':')))

# weekly units: same panel as notebook part C
c = comb[comb.order_item_id.notna() & comb.product_category_name_english.notna()].copy()
c['week'] = c.order_purchase_timestamp.dt.to_period('W').dt.start_time
c = c[(c.week >= '2017-01-16') & (c.week <= '2018-08-20')]
TOP = c.product_category_name_english.value_counts().head(20).index
u = c[c.product_category_name_english.isin(TOP)].groupby(['week', 'product_category_name_english']).size().unstack(fill_value=0)
u = u.reindex(pd.date_range('2017-01-16', '2018-08-20', freq='W-MON'), fill_value=0)[TOP]; u.index.name = 'week'
u.to_csv(root / 'data' / 'weekly_units.csv')

# sample "shop" orders: 150 delivered orders from the a 45-day window, with their real outcome (for the portal and order list)
# window ends 60 days before the last purchase so that (almost) every order of it is already delivered; the newest orders would be biased towards fast ones
tmax = dl.order_purchase_timestamp.max(); s = dl[(dl.order_purchase_timestamp >= tmax - pd.Timedelta(days=105)) & (dl.order_purchase_timestamp <= tmax - pd.Timedelta(days=60))]
s = s.sample(150, random_state=7).sort_values('order_purchase_timestamp')
rows = []
for r in s.itertuples():
    t = r.order_purchase_timestamp; hist = dl[dl.order_delivered_customer_date < t]
    w14 = hist[hist.order_delivered_customer_date >= t - pd.Timedelta(days=14)].days.mean()
    w14 = w14 if w14 == w14 else g_recent
    def rm(mask, val, win, k, prior, known='order_delivered_customer_date'):
        h = hist[mask(hist)] if mask else hist; h = h[h[known] >= t - pd.Timedelta(days=win)]
        return float((h[val].sum() + k * prior) / (len(h) + k))
    inp = {'seller_id': r.seller_id, 'seller_state': r.seller_state, 'customer_state': r.customer_state, 'freight': float(r.freight), 'weight_g': float(r.weight_g),
           'n_items': int(r.n_items), 'n_sellers': int(r.n_sellers), 'payment_type': r.payment_type if isinstance(r.payment_type, str) else 'credit_card',
           'est_days': float(r.est_days), 'limit_days': float(r.limit_days), 's_lat': float(r.s_lat),
           'recent_route_days': rm(lambda h: (h.seller_state == r.seller_state) & (h.customer_state == r.customer_state), 'days', 45, 5, w14),
           'recent_dest_days': rm(lambda h: h.customer_state == r.customer_state, 'days', 21, 5, w14),
           'recent_seller_carrier': rm(lambda h: h.seller_id == r.seller_id, 'carrier_days', 60, 3, gl['carrier_days'], 'order_delivered_carrier_date'),
           'seller_backlog': 0.0}
    rows.append({'order_id': r.order_id, 'purchase_time': str(t), 'category': r.category if isinstance(r.category, str) else 'unknown',
                 'price': float(r.price), 'input': inp, 'actual_days': round(float(r.days), 1), 'carrier_days': round(float(r.carrier_days), 1),
                 'approved': str(r.order_approved_at) if pd.notna(r.order_approved_at) else None,
                 'handed_to_carrier': str(r.order_delivered_carrier_date), 'delivered': str(r.order_delivered_customer_date),
                 'marketplace_estimate_days': round(float(r.est_days), 1)})
(root / 'data' / 'sample_orders.json').write_text(json.dumps(rows, separators=(',', ':')))
print('reference as of', ASOF.date(), '|', len(routes), 'routes |', len(sellers), 'sellers |', u.shape, 'weekly panel |', len(rows), 'sample orders')
