import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = import.meta.env.VITE_API_URL;

function money(v) { return `£${Number(v || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`; }
function pct(v) { return `${(Number(v || 0) * 100).toFixed(1)}%`; }

function App() {
  const [customers, setCustomers] = useState([]);
  const [form, setForm] = useState(null);
  const [result, setResult] = useState(null);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/customers`).then(r => r.json()),
      fetch(`${API}/training-summary`).then(r => r.json())
    ]).then(([customerData, trainingData]) => {
      setCustomers(customerData.customers || []);
      setSummary(trainingData);
      if (customerData.customers?.length) setForm(customerData.customers[0]);
    }).catch(() => setError('Could not connect to backend. Make sure FastAPI is running on port 9002.'));
  }, []);

  const selectCustomer = (id) => {
    const c = customers.find(x => x.customer_id === id);
    setForm(c);
    setResult(null);
  };

  const update = (field, value) => setForm({ ...form, [field]: value });

  const assess = async () => {
    setError('');
    setResult(null);
    try {
      const res = await fetch(`${API}/assess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          age: Number(form.age), annual_income: Number(form.annual_income), monthly_expenses: Number(form.monthly_expenses),
          years_employed: Number(form.years_employed), current_credit_limit: Number(form.current_credit_limit),
          requested_credit_increase: Number(form.requested_credit_increase), declared_existing_debt: Number(form.declared_existing_debt),
          declared_monthly_repayment: Number(form.declared_monthly_repayment), credit_score: Number(form.credit_score),
          utilisation_ratio: Number(form.utilisation_ratio), previous_defaults: Number(form.previous_defaults),
          missed_payments_12m: Number(form.missed_payments_12m), on_time_payment_rate: Number(form.on_time_payment_rate),
          arrears_balance: Number(form.arrears_balance)
        })
      });
      setResult(await res.json());
    } catch {
      setError('Assessment failed. Check backend terminal for errors.');
    }
  };

  if (error) return <div className="page"><div className="error">{error}</div></div>;
  if (!form) return <div className="page">Loading credit risk app...</div>;

  return <div className="page">
    <header className="hero">
      <div>
        <p className="eyebrow">AI Credit Risk Decision Engine</p>
        <h1>Credit Limit Increase Assessment</h1>
        <p>Select a customer, adjust the data, then run the ML risk assessment.</p>
      </div>
      <div className="modelBox">
        <b>ML Model</b><br />Random Forest<br />
        Training rows: {summary?.rows || 500}<br />
        Synthetic default rate: {summary ? pct(summary.default_rate) : '—'}
      </div>
    </header>

    <section className="grid">
      <div className="card">
        <h2>1. Select customer</h2>
        <select value={form.customer_id} onChange={e => selectCustomer(e.target.value)}>
          {customers.map(c => <option key={c.customer_id} value={c.customer_id}>{c.name} — {c.customer_id}</option>)}
        </select>

        <h2>2. Customer application</h2>
        <label>Name<input value={form.name} onChange={e => update('name', e.target.value)} /></label>
        <label>Employment status<input value={form.employment_status} onChange={e => update('employment_status', e.target.value)} /></label>
        <label>Annual income (£)<input type="number" value={form.annual_income} onChange={e => update('annual_income', e.target.value)} /></label>
        <label>Monthly expenses (£)<input type="number" value={form.monthly_expenses} onChange={e => update('monthly_expenses', e.target.value)} /></label>
        <label>Requested credit increase (£)<input type="number" value={form.requested_credit_increase} onChange={e => update('requested_credit_increase', e.target.value)} /></label>
        <label>Existing debt (£)<input type="number" value={form.declared_existing_debt} onChange={e => update('declared_existing_debt', e.target.value)} /></label>
      </div>

      <div className="card">
        <h2>3. Bureau + lender data</h2>
        <label>Credit score<input type="number" value={form.credit_score} onChange={e => update('credit_score', e.target.value)} /></label>
        <label>Utilisation ratio<input type="number" step="0.01" value={form.utilisation_ratio} onChange={e => update('utilisation_ratio', e.target.value)} /></label>
        <label>Previous defaults<input type="number" value={form.previous_defaults} onChange={e => update('previous_defaults', e.target.value)} /></label>
        <label>Missed payments in 12 months<input type="number" value={form.missed_payments_12m} onChange={e => update('missed_payments_12m', e.target.value)} /></label>
        <label>On-time payment rate<input type="number" step="0.01" value={form.on_time_payment_rate} onChange={e => update('on_time_payment_rate', e.target.value)} /></label>
        <label>Arrears balance (£)<input type="number" value={form.arrears_balance} onChange={e => update('arrears_balance', e.target.value)} /></label>
        <button onClick={assess}>Run ML Credit Assessment</button>
      </div>
    </section>

    {result && <section className="result">
      <div className={`decision ${result.risk_band.toLowerCase()}`}>{result.decision}<span>{result.risk_band} RISK</span></div>
      <div className="metrics">
        <div><b>PD</b><p>{pct(result.pd)}</p></div>
        <div><b>Expected Loss</b><p>{money(result.expected_loss)}</p></div>
        <div><b>Recommended Increase</b><p>{money(result.recommended_credit_increase)}</p></div>
        <div><b>Disposable Income</b><p>{money(result.disposable_income)}</p></div>
        <div><b>DTI</b><p>{pct(result.dti)}</p></div>
        <div><b>Stress PD</b><p>{pct(result.stressed_pd)}</p></div>
      </div>
      <div className="two">
        <div className="card"><h3>Decision reason</h3><p>{result.reason}</p><h3>Positive drivers</h3>{result.positive_drivers.map(x => <p className="good" key={x}>✓ {x}</p>)}</div>
        <div className="card"><h3>Early warning flags</h3>{result.early_warning_flags.length ? result.early_warning_flags.map(x => <p className="bad" key={x}>⚠ {x}</p>) : <p className="good">No major warning flags.</p>}</div>
      </div>
    </section>}

    <section className="card glossary">
      <h2>Risk glossary</h2>
      <p><b>Utilisation Ratio:</b> used credit divided by available credit limit. High utilisation means the customer is using a large share of available credit.</p>
      <p><b>PD:</b> Probability of Default — the model estimate that the customer may default.</p>
      <p><b>LGD:</b> Loss Given Default — how much the lender may lose if default happens.</p>
      <p><b>EAD:</b> Exposure at Default — the amount at risk when default happens.</p>
      <p><b>Expected Loss:</b> PD × LGD × EAD.</p>
    </section>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
