import React, { useEffect, useState } from "react";
import logo from "./logo.svg";
import "./App.css";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Scatter } from "recharts";

// TradeChart component for visualizing trade prices and actions
function TradeChart({ trades }) {
  // Prepare chart data: one point per trade
  const chartData = trades.map((t, idx) => ({
    ...t,
    idx,
    actionColor: t.action === "BUY" ? "#4caf50" : "#f44336"
  }));
  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis domain={[dataMin => Math.floor(dataMin * 0.95), dataMax => Math.ceil(dataMax * 1.05)]} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="price" stroke="#8884d8" dot={false} name="Price" />
        <Scatter dataKey="price" fill="#8884d8">
          {chartData.map((entry, idx) => (
            <circle
              key={idx}
              cx={null}
              cy={null}
              r={6}
              fill={entry.action === "BUY" ? "#4caf50" : "#f44336"}
              stroke="#222"
              strokeWidth={2}
            />
          ))}
        </Scatter>
      </LineChart>
    </ResponsiveContainer>
  );
}


function App() {
  const [parameters, setParameters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/parameters")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch parameters");
        return res.json();
      })
      .then((data) => {
        setParameters(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const [form, setForm] = useState({});
  const [result, setResult] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [mode, setMode] = useState("backtest"); // "backtest" or "optimize"
  const [optParams, setOptParams] = useState([]);

  useEffect(() => {
    // Initialize form with default values when parameters load
    if (parameters.length > 0) {
      const initial = {};
      parameters.forEach((p) => {
        initial[p.name] = p.default;
      });
      setForm(initial);
      setOptParams([]);
    }
  }, [parameters]);

  const handleChange = (name, type, value, checked) => {
    setForm((f) => ({
      ...f,
      [name]: type === "bool" ? checked : value,
    }));
  };

  const handleOptParamChange = (name, checked) => {
    setOptParams((prev) =>
      checked ? [...prev, name] : prev.filter((p) => p !== name)
    );
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmitting(true);
    setResult(null);
    const url = mode === "backtest" ? "http://localhost:8000/backtest" : "http://localhost:8000/optimize";
    const body = mode === "backtest"
      ? { parameters: form }
      : { parameters: form, optimize_params: optParams };
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`${mode === "backtest" ? "Backtest" : "Optimize"} failed`);
        return res.json();
      })
      .then((data) => {
        setResult(data);
        setSubmitting(false);
      })
      .catch((err) => {
        setResult({ error: err.message });
        setSubmitting(false);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <h2>Trading Parameters</h2>
        {loading && <p>Loading...</p>}
        {error && <p style={{ color: "red" }}>{error}</p>}
        {!loading && !error && (
          <>
            <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
              <button
                type="button"
                style={{
                  padding: "8px 16px",
                  marginRight: 8,
                  background: mode === "backtest" ? "#61dafb" : "#222",
                  color: mode === "backtest" ? "#222" : "#fff",
                  border: "none",
                  borderRadius: 4,
                  cursor: "pointer"
                }}
                onClick={() => setMode("backtest")}
              >
                Backtest
              </button>
              <button
                type="button"
                style={{
                  padding: "8px 16px",
                  background: mode === "optimize" ? "#61dafb" : "#222",
                  color: mode === "optimize" ? "#222" : "#fff",
                  border: "none",
                  borderRadius: 4,
                  cursor: "pointer"
                }}
                onClick={() => setMode("optimize")}
              >
                Optimize
              </button>
            </div>
            <form onSubmit={handleSubmit} style={{ textAlign: "left", margin: "0 auto", maxWidth: 400 }}>
              <h3>{mode === "backtest" ? "Edit Parameters" : "Optimize Parameters"}</h3>
              {parameters.map((param) => (
                <div key={param.name} style={{ marginBottom: 16 }}>
                  <label>
                    <strong>{param.name}</strong> ({param.type})<br />
                    {param.type === "bool" ? (
                      <input
                        type="checkbox"
                        checked={!!form[param.name]}
                        onChange={(e) => handleChange(param.name, param.type, e.target.value, e.target.checked)}
                      />
                    ) : (
                      <input
                        type={param.type === "int" || param.type === "float" ? "number" : "text"}
                        step={param.type === "float" ? "any" : "1"}
                        min={param.min !== null ? param.min : undefined}
                        max={param.max !== null ? param.max : undefined}
                        value={form[param.name] ?? ""}
                        onChange={(e) => handleChange(param.name, param.type, e.target.value)}
                      />
                    )}
                    {param.choices && param.type !== "bool" && (
                      <>
                        <br />Choices: {param.choices.join(", ")}
                      </>
                    )}
                  </label>
                  {mode === "optimize" && param.type !== "bool" && (
                    <label style={{ marginLeft: 16 }}>
                      <input
                        type="checkbox"
                        checked={optParams.includes(param.name)}
                        onChange={(e) => handleOptParamChange(param.name, e.target.checked)}
                      /> Optimize
                    </label>
                  )}
                </div>
              ))}
              <button type="submit" disabled={submitting} style={{ padding: "8px 16px" }}>
                {submitting ? (mode === "backtest" ? "Running Backtest..." : "Running Optimize...") : (mode === "backtest" ? "Run Backtest" : "Run Optimize")}
              </button>
            </form>
          </>
        )}
      </header>
       <section style={{ marginTop: 32 }}>
        {result && (
          result.error ? (
            <p style={{ color: "red" }}>Error: {result.error}</p>
          ) : (
            <div style={{ background: "#222", color: "#fff", padding: 24, borderRadius: 8, maxWidth: 700, margin: "0 auto" }}>
              <h3>{mode === "backtest" ? "Backtest Result" : "Optimize Result"}</h3>
              <p><strong>Profit/Loss:</strong> {result.profit_loss}</p>
              <h4>Stats</h4>
              <ul>
                {result.stats && Object.entries(result.stats).map(([k, v]) => (
                  <li key={k}><strong>{k}:</strong> {String(v)}</li>
                ))}
              </ul>
              {result.trades && result.trades.length > 0 && (
                <>
                  <h4>Trade Price Chart</h4>
                  <div style={{ background: "#fff", borderRadius: 8, padding: 16, marginBottom: 24 }}>
                    <TradeChart trades={result.trades} />
                  </div>
                </>
              )}
              <h4>Trades</h4>
              <table style={{ width: "100%", color: "#fff", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Action</th>
                    <th>Price</th>
                    <th>PnL</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trades && result.trades.map((trade, idx) => (
                    <tr key={idx}>
                      <td>{trade.date}</td>
                      <td>{trade.action}</td>
                      <td>{trade.price}</td>
                      <td>{trade.pnl}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        )}
      </section>

// --- TradeChart component ---

    </div>
  );
}

export default App;
