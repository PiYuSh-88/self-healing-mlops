import { useState, useEffect } from 'react'
import './index.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ── Default form values ──────────────────────────────────────────
const defaultForm = {
  gender: 'Male',
  SeniorCitizen: 0,
  Partner: 'Yes',
  Dependents: 'No',
  tenure: 12,
  PhoneService: 'Yes',
  MultipleLines: 'No',
  InternetService: 'DSL',
  OnlineSecurity: 'No',
  OnlineBackup: 'No',
  DeviceProtection: 'No',
  TechSupport: 'No',
  StreamingTV: 'No',
  StreamingMovies: 'No',
  Contract: 'Month-to-month',
  PaperlessBilling: 'Yes',
  PaymentMethod: 'Electronic check',
  MonthlyCharges: 50.0,
  TotalCharges: 600.0,
}

function App() {
  const [form, setForm] = useState(defaultForm)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)

  // ── Fetch health on mount ──────────────────────────────────────
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'unreachable' }))
  }, [])

  // ── Form change handler ────────────────────────────────────────
  const handleChange = (e) => {
    const { name, value } = e.target
    setForm(prev => ({
      ...prev,
      [name]: ['tenure', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges'].includes(name)
        ? Number(value) : value
    }))
  }

  // ── Prediction submit ──────────────────────────────────────────
  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to reach API')
    } finally {
      setLoading(false)
    }
  }

  // ── Select options helper ──────────────────────────────────────
  const yesNo = ['Yes', 'No']
  const yesNoService = ['Yes', 'No', 'No internet service']
  const yesNoPhone = ['Yes', 'No', 'No phone service']

  return (
    <div className="app">
      {/* ══ HEADER ══ */}
      <header className="header">
        <div className="header-left">
          <div className="header-icon">🧠</div>
          <h1>MLOps <span>Dashboard</span></h1>
        </div>
        <div className="header-badge">
          {health?.status === 'healthy' ? '● System Online' : '○ Connecting...'}
        </div>
      </header>

      {/* ══ STATUS CARDS ══ */}
      <div className="section-title"><span className="icon">📊</span> System Overview</div>
      <div className="status-grid">
        <div className={`status-card ${health?.status === 'healthy' ? 'healthy' : 'warning'}`}>
          <div className="status-card-header">
            <span className="status-card-label">API Status</span>
            <span className={`status-dot ${health?.status === 'healthy' ? 'green' : 'yellow'}`} />
          </div>
          <div className="status-card-value">
            {health?.status === 'healthy' ? 'Healthy' : 'Offline'}
          </div>
          <div className="status-card-desc">FastAPI Inference Engine</div>
        </div>

        <div className="status-card info">
          <div className="status-card-header">
            <span className="status-card-label">Model</span>
            <span className={`status-dot ${health?.model_loaded ? 'green' : 'yellow'}`} />
          </div>
          <div className="status-card-value">
            {health?.model_loaded ? 'Loaded' : '—'}
          </div>
          <div className="status-card-desc">RandomForest Classifier</div>
        </div>

        <div className="status-card accent">
          <div className="status-card-header">
            <span className="status-card-label">Predictions</span>
            <span className="status-dot purple" />
          </div>
          <div className="status-card-value">
            {health?.total_predictions_logged ?? '—'}
          </div>
          <div className="status-card-desc">Total predictions logged</div>
        </div>

        <div className="status-card healthy">
          <div className="status-card-header">
            <span className="status-card-label">Kubernetes</span>
            <span className="status-dot green" />
          </div>
          <div className="status-card-value">Running</div>
          <div className="status-card-desc">2 Pods · LoadBalancer</div>
        </div>
      </div>

      {/* ══ PREDICTION FORM + RESULT ══ */}
      <div className="main-grid">
        {/* ── Form Card ── */}
        <div className="card">
          <div className="card-title">🔮 Predict Customer Churn</div>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <FormSelect label="Gender" name="gender" value={form.gender} onChange={handleChange} options={['Male', 'Female']} />
              <FormSelect label="Senior Citizen" name="SeniorCitizen" value={form.SeniorCitizen} onChange={handleChange} options={[{v:0,l:'No'},{v:1,l:'Yes'}]} />
              <FormSelect label="Partner" name="Partner" value={form.Partner} onChange={handleChange} options={yesNo} />
              <FormSelect label="Dependents" name="Dependents" value={form.Dependents} onChange={handleChange} options={yesNo} />
              <FormInput label="Tenure (months)" name="tenure" type="number" value={form.tenure} onChange={handleChange} />
              <FormSelect label="Phone Service" name="PhoneService" value={form.PhoneService} onChange={handleChange} options={yesNo} />
              <FormSelect label="Multiple Lines" name="MultipleLines" value={form.MultipleLines} onChange={handleChange} options={yesNoPhone} />
              <FormSelect label="Internet Service" name="InternetService" value={form.InternetService} onChange={handleChange} options={['DSL', 'Fiber optic', 'No']} />
              <FormSelect label="Online Security" name="OnlineSecurity" value={form.OnlineSecurity} onChange={handleChange} options={yesNoService} />
              <FormSelect label="Online Backup" name="OnlineBackup" value={form.OnlineBackup} onChange={handleChange} options={yesNoService} />
              <FormSelect label="Device Protection" name="DeviceProtection" value={form.DeviceProtection} onChange={handleChange} options={yesNoService} />
              <FormSelect label="Tech Support" name="TechSupport" value={form.TechSupport} onChange={handleChange} options={yesNoService} />
              <FormSelect label="Streaming TV" name="StreamingTV" value={form.StreamingTV} onChange={handleChange} options={yesNoService} />
              <FormSelect label="Streaming Movies" name="StreamingMovies" value={form.StreamingMovies} onChange={handleChange} options={yesNoService} />
              <FormSelect label="Contract" name="Contract" value={form.Contract} onChange={handleChange} options={['Month-to-month', 'One year', 'Two year']} />
              <FormSelect label="Paperless Billing" name="PaperlessBilling" value={form.PaperlessBilling} onChange={handleChange} options={yesNo} />
              <FormSelect label="Payment Method" name="PaymentMethod" value={form.PaymentMethod} onChange={handleChange}
                options={['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']} />
              <FormInput label="Monthly Charges ($)" name="MonthlyCharges" type="number" value={form.MonthlyCharges} onChange={handleChange} step="0.01" />
              <FormInput label="Total Charges ($)" name="TotalCharges" type="number" value={form.TotalCharges} onChange={handleChange} step="0.01" />
            </div>
            <button type="submit" className="btn-predict" disabled={loading}>
              {loading ? <><span className="spinner" /> Predicting...</> : '🚀 Predict Churn'}
            </button>
          </form>
        </div>

        {/* ── Result Card ── */}
        <div className="card">
          <div className="card-title">📈 Prediction Result</div>
          {!result && !error && (
            <div className="result-placeholder">
              <div className="icon">📊</div>
              <p>Fill the form and click Predict</p>
            </div>
          )}
          {error && (
            <div className="error-msg">⚠️ {error}</div>
          )}
          {result && (
            <div className="result-content">
              <div className={`result-badge ${result.prediction === 'Churn' ? 'churn' : 'no-churn'}`}>
                {result.prediction === 'Churn' ? '🔴' : '🟢'}
                {result.prediction}
              </div>

              <div className="prob-section">
                <div className="prob-label">
                  <span>Churn Risk</span>
                  <span>{(result.churn_probability * 100).toFixed(1)}%</span>
                </div>
                <div className="prob-bar-bg">
                  <div className="prob-bar-fill churn" style={{ width: `${result.churn_probability * 100}%` }} />
                </div>
              </div>

              <div className="prob-section">
                <div className="prob-label">
                  <span>Retention Confidence</span>
                  <span>{(result.no_churn_probability * 100).toFixed(1)}%</span>
                </div>
                <div className="prob-bar-bg">
                  <div className="prob-bar-fill safe" style={{ width: `${result.no_churn_probability * 100}%` }} />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ══ ARCHITECTURE FLOW ══ */}
      <div className="section-title"><span className="icon">⚙️</span> Self-Healing Architecture</div>
      <div className="arch-flow">
        <ArchNode icon="👤" label="User" />
        <span className="arch-arrow">→</span>
        <ArchNode icon="⚛️" label="React UI" />
        <span className="arch-arrow">→</span>
        <ArchNode icon="⚡" label="FastAPI" />
        <span className="arch-arrow">→</span>
        <ArchNode icon="🧠" label="ML Model" />
        <span className="arch-arrow">→</span>
        <ArchNode icon="📊" label="Prometheus" />
        <span className="arch-arrow">→</span>
        <ArchNode icon="📈" label="Grafana" />
        <span className="arch-arrow">→</span>
        <ArchNode icon="🔄" label="Jenkins" />
        <span className="arch-arrow">→</span>
        <ArchNode icon="☸️" label="K8s" />
      </div>

      {/* ══ FOOTER ══ */}
      <footer className="footer">
        Self-Healing MLOps Platform · Built with FastAPI, Docker, Kubernetes, Jenkins, Prometheus & Grafana
      </footer>
    </div>
  )
}

// ── Reusable Components ────────────────────────────────────────────

function FormInput({ label, name, type, value, onChange, step }) {
  return (
    <div className="form-group">
      <label htmlFor={name}>{label}</label>
      <input id={name} name={name} type={type} value={value} onChange={onChange} step={step} />
    </div>
  )
}

function FormSelect({ label, name, value, onChange, options }) {
  return (
    <div className="form-group">
      <label htmlFor={name}>{label}</label>
      <select id={name} name={name} value={value} onChange={onChange}>
        {options.map(opt => {
          const val = typeof opt === 'object' ? opt.v : opt
          const lbl = typeof opt === 'object' ? opt.l : opt
          return <option key={val} value={val}>{lbl}</option>
        })}
      </select>
    </div>
  )
}

function ArchNode({ icon, label }) {
  return (
    <div className="arch-node">
      <span className="icon">{icon}</span>
      <span className="label">{label}</span>
    </div>
  )
}

export default App
