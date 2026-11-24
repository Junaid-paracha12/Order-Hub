"""FastAPI service + static frontend.   run:  uvicorn app.main:app --reload"""
from pathlib import Path
import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from . import pipeline as P

app = FastAPI(title='Olist Order Management Hub', version='1.0')
FRONT = P.ROOT / 'frontend'


class OrderIn(BaseModel):
    seller_state: str = Field(min_length=2, max_length=2, examples=['SP'])
    customer_state: str = Field(min_length=2, max_length=2, examples=['RJ'])
    freight: float | None = Field(None, ge=0, examples=[18.5], description='leave empty: typical freight for the route and weight')
    weight_g: float = Field(gt=0, examples=[1200])
    n_items: int = Field(1, ge=1, le=30)
    n_sellers: int = Field(1, ge=1, le=10)
    payment_type: str = 'credit_card'
    seller_id: str | None = None
    purchase_time: str | None = Field(None, examples=['2018-08-20 14:30'])
    est_days: float | None = Field(None, description="the marketplace's own delivery estimate in days (default: typical for the route)")
    desired_date: str | None = Field(None, examples=['2018-09-05'], description='date the customer wants the parcel by (YYYY-MM-DD)')
    on_time_target: float = Field(0.94, ge=0.5, le=0.99)


class StockIn(BaseModel):
    stock: dict[str, float]
    week: str | None = None
    service_level: float = Field(0.90, ge=0.5, le=0.99)


@app.get('/api/health')
def health(): return {'status': 'ok', 'reference_as_of': P.reference()['asof']}


@app.get('/api/meta')
def meta():
    r = P.reference()
    return {'customer_states': r['customer_states'], 'seller_states': r['seller_states'], 'payment_types': r['payment_types'],
            'categories': P.CATS_SORTED(), 'last_demand_week': str(P.weekly_units().index.max().date()), 'top_sellers': list(r['sellers'])[:30]}


@app.post('/api/delivery/predict')
def delivery_predict(o: OrderIn):
    d = o.model_dump(exclude_none=True); tgt = d.pop('on_time_target')
    r = P.reference()
    if d['seller_state'] not in r['states'] or d['customer_state'] not in r['states']: raise HTTPException(422, 'unknown state code')
    if d['payment_type'] not in r['payment_types']: d['payment_type'] = 'unknown'
    return P.predict_delivery(d, tgt)


@app.get('/api/freight')
def freight(seller_state: str, customer_state: str, weight_g: float = Query(1000, gt=0)):
    r = P.reference()
    if seller_state not in r['states'] or customer_state not in r['states']: raise HTTPException(422, 'unknown state code')
    return {'freight': round(P.estimate_freight(seller_state, customer_state, weight_g), 2)}


@app.get('/api/orders')
def orders(limit: int = Query(60, le=150)):
    out = []
    for o in P.sample_orders()[-limit:][::-1]:
        p = P.predict_delivery({**o['input'], 'purchase_time': o['purchase_time']})
        out.append({'order_id': o['order_id'], 'purchase_time': o['purchase_time'], 'category': o['category'], 'price': o['price'],
                    'route': f"{o['input']['seller_state']} -> {o['input']['customer_state']}", 'expected_days': p['expected_days'], 'promise_days': p['promise_days'],
                    'marketplace_estimate_days': p['marketplace_estimate_days'], 'late_risk': p['late_risk'], 'alert': p['alert'], 'actual_days': o['actual_days']})
    return out


@app.get('/api/orders/{order_id}')
def portal(order_id: str):
    v = P.portal_view(order_id)
    if not v: raise HTTPException(404, 'order not found')
    return v


@app.get('/api/demand')
def demand(week: str | None = None, service_level: float = Query(0.9, ge=0.5, le=0.99)):
    try: return P.forecast_demand(week, service_level)
    except ValueError as e: raise HTTPException(422, str(e))


@app.post('/api/demand/stock')
def demand_stock(b: StockIn):
    return P.stock_actions(P.forecast_demand(b.week, b.service_level), b.stock)


@app.get('/api/demand/history')
def history(category: str, weeks: int = Query(26, le=84)):
    u = P.weekly_units()
    if category not in u.columns: raise HTTPException(404, 'unknown category')
    s = u[category].iloc[-weeks:]; return {'weeks': [str(i.date()) for i in s.index], 'units': [int(v) for v in s]}


@app.get('/api/report')
def report(): return json.loads((P.ROOT / 'data' / 'report.json').read_text())


@app.get('/')
def index(): return FileResponse(FRONT / 'index.html')


app.mount('/static', StaticFiles(directory=FRONT), name='static')
