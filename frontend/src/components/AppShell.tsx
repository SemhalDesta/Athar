import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "../api";
import { useSettings } from "../settings";
import { Brand } from "./Brand";

export default function AppShell() {
  const { settings, setSettings } = useSettings();
  const loc = useLocation();
  const [health, setHealth] = useState<Awaited<ReturnType<typeof api.health>> | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { api.health().then(setHealth).catch((e) => setErr(e.message)); }, []);
  const isHome = loc.pathname === "/";

  return (
    <div className={`app ${isHome ? "is-home" : ""}`}>
      <header className="app-top">
        <Brand to="/" />
        <nav className="app-tabs" aria-label="Main">
          <NavLink to="/" end>Home</NavLink>
          <NavLink to="/review">Patient review</NavLink>
          <NavLink to="/performance">Performance</NavLink>
          <NavLink to="/how-it-works">How it works</NavLink>
        </nav>
        <div className="app-right">
          <details className="demo-settings">
            <summary>Demo settings</summary>
            <div className="demo-pop">
              <label className="ctl">Patients
                <select value={settings.dataset} onChange={(e) => setSettings({ ...settings, dataset: e.target.value })}>
                  <option value="eval">Evaluation set (held out)</option>
                  <option value="dev">Development set</option>
                </select>
              </label>
              <p className="small muted">All patients are fictional. Notes read by {health ? (health.extractor_by_dataset[settings.dataset] === "llm" ? `the LLM (${health.llm_model})` : "the keyword extractor") : "…"}.</p>
            </div>
          </details>
        </div>
      </header>
      {err && !isHome && <div className="notice-err">Cannot reach the Athar API ({err}). Start it in the backend folder with <code>py -m uvicorn athar.api.main:app --port 8000</code>, then refresh.</div>}
      <main className={isHome ? "home-main" : loc.pathname.startsWith("/review") ? "ws-main" : "content"}><Outlet /></main>
    </div>
  );
}
