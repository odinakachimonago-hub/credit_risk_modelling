from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="AI Credit Risk Decision Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002","https://eclectic-cheesecake-a7a016.netlify.app","https://riskmodelling.netlify.app", "http://127.0.0.1:3002", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CUSTOMERS = [
    {"customer_id":"CUST001","name":"Sarah Johnson","age":32,"annual_income":45000,"monthly_expenses":1600,"employment_status":"Full-time","years_employed":4,"current_credit_limit":3000,"requested_credit_increase":5000,"loan_purpose":"Credit limit increase","declared_existing_debt":8000,"declared_monthly_repayment":350,"credit_score":690,"utilisation_ratio":0.35,"previous_defaults":0,"missed_payments_12m":1,"on_time_payment_rate":0.97,"arrears_balance":0},
    {"customer_id":"CUST002","name":"Michael Brown","age":41,"annual_income":38000,"monthly_expenses":1900,"employment_status":"Full-time","years_employed":6,"current_credit_limit":2500,"requested_credit_increase":6000,"loan_purpose":"Credit limit increase","declared_existing_debt":18000,"declared_monthly_repayment":900,"credit_score":580,"utilisation_ratio":0.82,"previous_defaults":1,"missed_payments_12m":4,"on_time_payment_rate":0.78,"arrears_balance":250},
    {"customer_id":"CUST003","name":"Aisha Khan","age":29,"annual_income":62000,"monthly_expenses":2100,"employment_status":"Full-time","years_employed":5,"current_credit_limit":5000,"requested_credit_increase":4000,"loan_purpose":"Credit limit increase","declared_existing_debt":4000,"declared_monthly_repayment":210,"credit_score":750,"utilisation_ratio":0.22,"previous_defaults":0,"missed_payments_12m":0,"on_time_payment_rate":1.0,"arrears_balance":0},
    {"customer_id":"CUST004","name":"Daniel Evans","age":36,"annual_income":42000,"monthly_expenses":1850,"employment_status":"Self-employed","years_employed":3,"current_credit_limit":3000,"requested_credit_increase":5000,"loan_purpose":"Credit limit increase","declared_existing_debt":12000,"declared_monthly_repayment":620,"credit_score":640,"utilisation_ratio":0.60,"previous_defaults":0,"missed_payments_12m":2,"on_time_payment_rate":0.88,"arrears_balance":90},
    {"customer_id":"CUST005","name":"Priya Patel","age":27,"annual_income":52000,"monthly_expenses":1750,"employment_status":"Full-time","years_employed":2,"current_credit_limit":4000,"requested_credit_increase":3500,"loan_purpose":"Credit limit increase","declared_existing_debt":6000,"declared_monthly_repayment":310,"credit_score":710,"utilisation_ratio":0.41,"previous_defaults":0,"missed_payments_12m":0,"on_time_payment_rate":0.98,"arrears_balance":0},
    {"customer_id":"CUST006","name":"James Wilson","age":48,"annual_income":72000,"monthly_expenses":2800,"employment_status":"Full-time","years_employed":12,"current_credit_limit":8000,"requested_credit_increase":6000,"loan_purpose":"Credit limit increase","declared_existing_debt":15000,"declared_monthly_repayment":760,"credit_score":705,"utilisation_ratio":0.52,"previous_defaults":0,"missed_payments_12m":1,"on_time_payment_rate":0.96,"arrears_balance":0},
    {"customer_id":"CUST007","name":"Chloe Martin","age":24,"annual_income":29000,"monthly_expenses":1450,"employment_status":"Part-time","years_employed":1,"current_credit_limit":1800,"requested_credit_increase":3000,"loan_purpose":"Credit limit increase","declared_existing_debt":7000,"declared_monthly_repayment":420,"credit_score":625,"utilisation_ratio":0.68,"previous_defaults":0,"missed_payments_12m":2,"on_time_payment_rate":0.89,"arrears_balance":60},
    {"customer_id":"CUST008","name":"Omar Ali","age":39,"annual_income":83000,"monthly_expenses":3100,"employment_status":"Self-employed","years_employed":8,"current_credit_limit":10000,"requested_credit_increase":9000,"loan_purpose":"Credit limit increase","declared_existing_debt":18000,"declared_monthly_repayment":850,"credit_score":735,"utilisation_ratio":0.33,"previous_defaults":0,"missed_payments_12m":0,"on_time_payment_rate":0.99,"arrears_balance":0},
    {"customer_id":"CUST009","name":"Emily Taylor","age":31,"annual_income":34000,"monthly_expenses":1700,"employment_status":"Unemployed","years_employed":0,"current_credit_limit":2200,"requested_credit_increase":2000,"loan_purpose":"Credit limit increase","declared_existing_debt":9000,"declared_monthly_repayment":520,"credit_score":595,"utilisation_ratio":0.76,"previous_defaults":1,"missed_payments_12m":3,"on_time_payment_rate":0.80,"arrears_balance":180},
    {"customer_id":"CUST010","name":"Noah Clarke","age":44,"annual_income":56000,"monthly_expenses":2050,"employment_status":"Full-time","years_employed":9,"current_credit_limit":6000,"requested_credit_increase":5000,"loan_purpose":"Credit limit increase","declared_existing_debt":9500,"declared_monthly_repayment":430,"credit_score":680,"utilisation_ratio":0.49,"previous_defaults":0,"missed_payments_12m":1,"on_time_payment_rate":0.95,"arrears_balance":0},
]

FEATURES = ["credit_score","previous_defaults","missed_payments_12m","utilisation_ratio","declared_existing_debt","declared_monthly_repayment","years_employed","on_time_payment_rate","arrears_balance"]

def make_training_data(n=500):
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n):
        credit_score = int(rng.integers(520, 801))
        previous_defaults = int(rng.choice([0, 0, 0, 1], p=[0.55, 0.2, 0.1, 0.15]))
        missed = int(rng.integers(0, 6))
        util = float(np.round(rng.uniform(0.1, 0.95), 2))
        debt = float(rng.integers(1500, 30000))
        repayment = float(rng.integers(100, 1500))
        years = float(rng.integers(0, 15))
        pay_rate = float(np.round(rng.uniform(0.65, 1.0), 2))
        arrears = float(rng.choice([0, 0, 0, 50, 150, 400, 900]))
        risk_score = (
            (700 - credit_score) / 180
            + previous_defaults * 0.9
            + missed * 0.18
            + util * 0.8
            + debt / 45000
            + repayment / 2500
            - years * 0.03
            - pay_rate * 0.7
            + arrears / 1400
        )
        prob_default = 1 / (1 + np.exp(-2.0 * (risk_score - 0.9)))
        defaulted = int(rng.random() < prob_default)
        rows.append([credit_score, previous_defaults, missed, util, debt, repayment, years, pay_rate, arrears, defaulted])
    return pd.DataFrame(rows, columns=FEATURES + ["defaulted"])

TRAINING_DATA = make_training_data(500)
MODEL = RandomForestClassifier(n_estimators=120, random_state=42, max_depth=6)
MODEL.fit(TRAINING_DATA[FEATURES], TRAINING_DATA["defaulted"])

class CreditRequest(BaseModel):
    customer_id: str
    name: str
    age: int
    annual_income: float
    monthly_expenses: float
    employment_status: str
    years_employed: float
    current_credit_limit: float
    requested_credit_increase: float
    loan_purpose: str = "Credit limit increase"
    declared_existing_debt: float
    declared_monthly_repayment: float
    credit_score: int
    utilisation_ratio: float
    previous_defaults: int
    missed_payments_12m: int
    on_time_payment_rate: float
    arrears_balance: float

@app.get("/")
def root():
    return {"message": "AI Credit Risk Decision Engine API is running", "try": "/docs"}

@app.get("/customers")
def customers():
    return {"customers": CUSTOMERS}

@app.get("/training-summary")
def training_summary():
    return {
        "rows": int(len(TRAINING_DATA)),
        "default_rate": round(float(TRAINING_DATA["defaulted"].mean()), 3),
        "features": FEATURES,
        "model": "Random Forest Classifier"
    }

@app.post("/assess")
def assess(req: CreditRequest):
    monthly_income = req.annual_income / 12
    disposable_income = monthly_income - req.monthly_expenses - req.declared_monthly_repayment
    dti = req.declared_existing_debt / req.annual_income if req.annual_income else 0
    new_limit = req.current_credit_limit + req.requested_credit_increase
    estimated_used_credit = req.utilisation_ratio * req.current_credit_limit
    post_increase_utilisation = estimated_used_credit / new_limit if new_limit else 0

    feature_row = pd.DataFrame([{
        "credit_score": req.credit_score,
        "previous_defaults": req.previous_defaults,
        "missed_payments_12m": req.missed_payments_12m,
        "utilisation_ratio": req.utilisation_ratio,
        "declared_existing_debt": req.declared_existing_debt,
        "declared_monthly_repayment": req.declared_monthly_repayment,
        "years_employed": req.years_employed,
        "on_time_payment_rate": req.on_time_payment_rate,
        "arrears_balance": req.arrears_balance,
    }])
    pd_score = float(MODEL.predict_proba(feature_row)[0][1])
    lgd = 0.45
    ead = new_limit
    expected_loss = pd_score * lgd * ead
    stressed_pd = min(pd_score * 1.6 + 0.03, 0.99)
    stressed_expected_loss = stressed_pd * lgd * ead

    affordability_cap = max(disposable_income * 6, 0)
    risk_cap = max((1 - pd_score) * req.current_credit_limit * 1.5, 0)
    recommended_increase = round(min(req.requested_credit_increase, affordability_cap, risk_cap), 2)

    flags = []
    positives = []
    if req.credit_score < 620: flags.append("Low credit score")
    else: positives.append("Acceptable/strong credit score")
    if req.previous_defaults > 0: flags.append("Previous default history")
    if req.missed_payments_12m >= 2: flags.append("Multiple missed payments in last 12 months")
    if req.utilisation_ratio > 0.65: flags.append("High utilisation ratio")
    if req.arrears_balance > 0: flags.append("Customer currently has arrears")
    if disposable_income < 400: flags.append("Low disposable income after expenses and repayments")
    if req.on_time_payment_rate >= 0.95: positives.append("Strong on-time payment rate")
    if req.years_employed >= 3: positives.append("Stable employment history")

    if pd_score < 0.18 and disposable_income > 500 and req.previous_defaults == 0 and req.arrears_balance == 0:
        decision, risk_band = "APPROVE", "LOW"
        reason = "Low predicted default risk, acceptable affordability and clean arrears/default profile."
    elif pd_score < 0.35 and disposable_income > 250:
        decision, risk_band = "REFER", "MEDIUM"
        reason = "Moderate model risk or affordability pressure. Manual credit review recommended."
    else:
        decision, risk_band = "DECLINE", "HIGH"
        reason = "High predicted default risk, weak affordability or adverse credit behaviour."

    return {
        "customer_id": req.customer_id,
        "name": req.name,
        "decision": decision,
        "risk_band": risk_band,
        "reason": reason,
        "pd": round(pd_score, 4),
        "lgd": lgd,
        "ead": round(ead, 2),
        "expected_loss": round(expected_loss, 2),
        "stressed_pd": round(stressed_pd, 4),
        "stressed_expected_loss": round(stressed_expected_loss, 2),
        "recommended_credit_increase": recommended_increase,
        "monthly_income": round(monthly_income, 2),
        "disposable_income": round(disposable_income, 2),
        "dti": round(dti, 4),
        "post_increase_utilisation": round(post_increase_utilisation, 4),
        "early_warning_flags": flags,
        "positive_drivers": positives,
        "definitions": {
            "PD": "Probability of Default: estimated chance the customer defaults.",
            "LGD": "Loss Given Default: percentage of exposure expected to be lost if default happens.",
            "EAD": "Exposure at Default: estimated amount outstanding when default happens.",
            "Expected Loss": "PD × LGD × EAD.",
            "Utilisation Ratio": "Used credit divided by available credit limit.",
            "DTI": "Debt-to-income ratio: existing debt divided by annual income."
        }
    }
