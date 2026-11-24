import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, pytest
from fastapi.testclient import TestClient
from app.main import app
from app import pipeline as P

c = TestClient(app)
ORDER = {'seller_state': 'SP', 'customer_state': 'RJ', 'freight': 20, 'weight_g': 1500, 'purchase_time': '2018-08-20 14:30'}


def test_health_and_meta():
    assert c.get('/api/health').json()['status'] == 'ok'
    m = c.get('/api/meta').json(); assert len(m['categories']) == 20 and 'SP' in m['customer_states']


def test_delivery_prediction_is_sane():
    r = c.post('/api/delivery/predict', json=ORDER).json()
    assert 1 < r['expected_days'] < 60 and r['promise_days'] >= r['expected_days'] and 0 <= r['late_risk'] <= 1 and 'promise_date' in r


def test_longer_target_never_shortens_promise():
    a = c.post('/api/delivery/predict', json={**ORDER, 'on_time_target': .8}).json()['promise_days']
    b = c.post('/api/delivery/predict', json={**ORDER, 'on_time_target': .98}).json()['promise_days']
    assert b >= a


def test_far_route_is_slower_than_local():
    near = c.post('/api/delivery/predict', json={**ORDER, 'customer_state': 'SP'}).json()['expected_days']
    far = c.post('/api/delivery/predict', json={**ORDER, 'customer_state': 'AM'}).json()['expected_days']
    assert far > near


def test_validation_errors():
    assert c.post('/api/delivery/predict', json={**ORDER, 'freight': -1}).status_code == 422
    assert c.post('/api/delivery/predict', json={**ORDER, 'customer_state': 'ZZ'}).status_code == 422


def test_demand_matches_notebook_next_week():
    f = {x['category']: x for x in c.get('/api/demand').json()}
    assert f['health_beauty']['week'] == '2018-08-27'
    assert f['health_beauty']['forecast_units'] == pytest.approx(155.4, abs=0.2) and f['health_beauty']['reorder_level'] == 233


def test_demand_uses_only_the_past():
    """Features of week t must not change if the data from week t onward is removed."""
    wk = pd.Timestamp('2018-03-05'); full = P.demand_features(wk)
    orig = P.weekly_units.__wrapped__
    P.weekly_units.cache_clear()
    try:
        cut = orig()[lambda d: d.index < wk]; P.weekly_units = lambda: cut
        assert np.allclose(full.drop(columns='cat').to_numpy(float), P.demand_features(wk).drop(columns='cat').to_numpy(float))
    finally:
        P.weekly_units = P.lru_cache(maxsize=1)(orig)


def test_higher_service_level_means_higher_reorder_level():
    a = {x['category']: x['reorder_level'] for x in c.get('/api/demand', params={'service_level': .8}).json()}
    b = {x['category']: x['reorder_level'] for x in c.get('/api/demand', params={'service_level': .95}).json()}
    assert all(b[k] >= a[k] for k in a)


def test_stock_actions():
    r = c.post('/api/demand/stock', json={'stock': {'health_beauty': 10, 'auto': 10_000}}).json()
    d = {x['category']: x for x in r}
    assert d['health_beauty']['status'] == 'REORDER' and d['health_beauty']['order_qty'] > 0 and d['auto']['status'] == 'OK' and d['auto']['order_qty'] == 0


def test_orders_and_portal():
    lst = c.get('/api/orders').json(); assert len(lst) == 60
    one = c.get(f"/api/orders/{lst[0]['order_id']}").json(); assert one['prediction']['promise_days'] > 0 and 'delivered' in one
    assert c.get('/api/orders/nope').status_code == 404


def test_sample_orders_beat_median_baseline():
    S = P.sample_orders(); a = np.array([o['actual_days'] for o in S])
    e = np.array([P.predict_delivery({**o['input'], 'purchase_time': o['purchase_time']})['expected_days'] for o in S])
    assert np.abs(a - e).mean() < np.abs(a - np.median(a)).mean()


def test_history_and_report():
    h = c.get('/api/demand/history', params={'category': 'auto', 'weeks': 10}).json(); assert len(h['units']) == 10
    assert c.get('/api/demand/history', params={'category': 'x'}).status_code == 404
    assert 'delivery' in c.get('/api/report').json()


def test_freight_estimate_grows_with_distance_and_weight():
    near = c.get('/api/freight', params={'seller_state': 'SP', 'customer_state': 'SP', 'weight_g': 800}).json()['freight']
    far = c.get('/api/freight', params={'seller_state': 'SP', 'customer_state': 'AM', 'weight_g': 800}).json()['freight']
    heavy = c.get('/api/freight', params={'seller_state': 'SP', 'customer_state': 'SP', 'weight_g': 20000}).json()['freight']
    assert far > near and heavy > near


def test_freight_is_optional_and_estimated():
    r = c.post('/api/delivery/predict', json={k: v for k, v in ORDER.items() if k != 'freight'}).json()
    assert r['freight_estimated'] and r['freight_used'] > 0
    assert not c.post('/api/delivery/predict', json=ORDER).json()['freight_estimated']


def test_desired_date_chance_rises_with_more_days():
    chances = []
    for d in ('2018-08-25', '2018-09-05', '2018-09-30'):
        r = c.post('/api/delivery/predict', json={**ORDER, 'desired_date': d}).json()['desired']; chances.append(r['chance'])
    assert chances == sorted(chances) and chances[0] < chances[-1]
    past = c.post('/api/delivery/predict', json={**ORDER, 'desired_date': '2018-08-19'}).json()['desired']
    assert past['verdict'] == 'impossible' and past['chance'] == 0
    assert c.post('/api/delivery/predict', json={**ORDER, 'desired_date': '2018-09-30'}).json()['desired']['verdict'] == 'safe'


def test_promise_is_calibrated_and_reasonable():
    r = c.post('/api/delivery/predict', json={**ORDER, 'customer_state': 'RJ'}).json()
    assert r['promise_days'] < 1.8 * r['expected_days'] + 3          # not the 2.4x blow-up of an uncalibrated quantile
    assert abs(P.on_time_for_ratio(P.ratio_for_on_time(0.94)) - 0.94) < 0.01
