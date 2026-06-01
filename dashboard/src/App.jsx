import { useState, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8000";
const POLL_MS  = 60_000; // 1 minute

const LABEL_META = {
  PHISHING:         { color: "#ff4444", bg: "#2a0a0a", icon: "⚠" },
  SAFE:             { color: "#00e676", bg: "#0a2a10", icon: "✓" },
  PROMPT_INJECTION: { color: "#ff9800", bg: "#2a1500", icon: "⚡" },
  REVIEW:           { color: "#ffeb3b", bg: "#2a2200", icon: "?" },
};

function LabelBadge({ label }) {
  const m = LABEL_META[label] || LABEL_META["REVIEW"];
  return (
    <span
      style={{
        background: m.bg,
        color: m.color,
        border: `1px solid ${m.color}44`,
        borderRadius: 4,
        padding: "2px 10px",
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 12,
        fontWeight: 700,
        letterSpacing: 1,
      }}
    >
      {m.icon} {label}
    </span>
  );
}

function StatCard({ label, count, color }) {
  return (
    <div
      style={{
        background: "#0d1117",
        border: `1px solid ${color}33`,
        borderRadius: 8,
        padding: "18px 24px",
        minWidth: 140,
        textAlign: "center",
      }}
    >
      <div style={{ color, fontSize: 32, fontWeight: 800, fontFamily: "monospace" }}>
        {count}
      </div>
      <div style={{ color: "#8b949e", fontSize: 12, marginTop: 4, letterSpacing: 1 }}>
        {label}
      </div>
    </div>
  );
}

export default function App() {
  const [results, setResults]     = useState([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [filterLabel, setFilter]  = useState("ALL");

  const fetchResults = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/results?limit=200`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setResults(data);
      setLastUpdate(new Date());
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchResults();
    const id = setInterval(fetchResults, POLL_MS);
    return () => clearInterval(id);
  }, [fetchResults]);

  // Stats
  const total     = results.length;
  const phishing  = results.filter(r => r.label === "PHISHING").length;
  const safe      = results.filter(r => r.label === "SAFE").length;
  const injection = results.filter(r => r.label === "PROMPT_INJECTION").length;
  const review    = results.filter(r => r.label === "REVIEW").length;

  const displayed = filterLabel === "ALL"
    ? results
    : results.filter(r => r.label === filterLabel);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#010409",
      color: "#e6edf3",
      fontFamily: "'Segoe UI', sans-serif",
      padding: "32px 24px",
    }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
          <span style={{ fontSize: 28 }}>🛡</span>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: "#58a6ff" }}>
            Email Security Dashboard
          </h1>
          <span style={{
            marginLeft: "auto",
            fontSize: 12,
            color: "#8b949e",
            fontFamily: "monospace",
          }}>
            {lastUpdate
              ? `Last updated: ${lastUpdate.toLocaleTimeString()}`
              : "Loading…"}
          </span>
          <button
            onClick={fetchResults}
            style={{
              background: "#1f2937",
              border: "1px solid #30363d",
              color: "#e6edf3",
              borderRadius: 6,
              padding: "4px 14px",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            ↻ Refresh
          </button>
        </div>
        <p style={{ margin: 0, color: "#8b949e", fontSize: 14 }}>
          Real-time phishing detection · Powered by Llama 3.2 + NeMo Guardrails · Auto-refreshes every 60s
        </p>
      </div>

      {/* Stat cards */}
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 28 }}>
        <StatCard label="TOTAL EMAILS"     count={total}     color="#58a6ff" />
        <StatCard label="PHISHING"         count={phishing}  color="#ff4444" />
        <StatCard label="SAFE"             count={safe}      color="#00e676" />
        <StatCard label="PROMPT INJECTION" count={injection} color="#ff9800" />
        <StatCard label="REVIEW"           count={review}    color="#ffeb3b" />
      </div>

      {/* Filter bar */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
        {["ALL", "PHISHING", "SAFE", "PROMPT_INJECTION", "REVIEW"].map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              background: filterLabel === f ? "#1f6feb" : "#161b22",
              border: `1px solid ${filterLabel === f ? "#388bfd" : "#30363d"}`,
              color: "#e6edf3",
              borderRadius: 6,
              padding: "5px 14px",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: filterLabel === f ? 700 : 400,
            }}
          >
            {f}
          </button>
        ))}
        <span style={{ marginLeft: "auto", fontSize: 12, color: "#8b949e", alignSelf: "center" }}>
          Showing {displayed.length} of {total}
        </span>
      </div>

      {/* Error state */}
      {error && (
        <div style={{
          background: "#2a0a0a",
          border: "1px solid #ff4444",
          borderRadius: 8,
          padding: "14px 20px",
          color: "#ff7070",
          marginBottom: 20,
          fontSize: 14,
        }}>
          ⚠ Cannot reach API: {error}. Make sure the inference server is running on port 8000.
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div style={{ color: "#8b949e", textAlign: "center", padding: 40 }}>
          Loading classifications…
        </div>
      )}

      {/* Results table */}
      {!loading && displayed.length === 0 && (
        <div style={{ color: "#8b949e", textAlign: "center", padding: 40 }}>
          No results yet. Emails will appear here once classified.
        </div>
      )}

      {!loading && displayed.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #21262d" }}>
                {["Timestamp", "Sender", "Subject", "Label", "Guardrail", "Latency"].map(h => (
                  <th key={h} style={{
                    textAlign: "left",
                    padding: "10px 14px",
                    fontSize: 12,
                    color: "#8b949e",
                    fontWeight: 600,
                    letterSpacing: 0.5,
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {displayed.map(row => (
                <tr
                  key={row.id}
                  style={{
                    borderBottom: "1px solid #161b22",
                    transition: "background 0.15s",
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = "#0d1117"}
                  onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                >
                  <td style={{ padding: "10px 14px", fontSize: 12, color: "#8b949e", fontFamily: "monospace", whiteSpace: "nowrap" }}>
                    {new Date(row.timestamp).toLocaleString()}
                  </td>
                  <td style={{ padding: "10px 14px", fontSize: 13, maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {row.sender || "—"}
                  </td>
                  <td style={{ padding: "10px 14px", fontSize: 13, maxWidth: 240, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {row.subject || "—"}
                  </td>
                  <td style={{ padding: "10px 14px" }}>
                    <LabelBadge label={row.label} />
                  </td>
                  <td style={{ padding: "10px 14px", fontSize: 12, color: "#8b949e" }}>
                    {row.guardrail_status}
                  </td>
                  <td style={{ padding: "10px 14px", fontSize: 12, color: "#8b949e", fontFamily: "monospace" }}>
                    {row.latency_ms?.toFixed(0)} ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
