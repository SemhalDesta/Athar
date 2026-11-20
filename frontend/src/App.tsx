import { useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Settings, SettingsContext } from "./settings";
import AppShell from "./components/AppShell";
import Home from "./components/Home";
import Workstation from "./components/Workstation";
import Performance from "./components/Performance";
import Method from "./components/Method";

export type { Settings };

export default function App() {
  const [settings, setSettings] = useState<Settings>({ dataset: "eval", showTruth: false });
  return (
    <SettingsContext.Provider value={{ settings, setSettings }}>
      <div className="aura" aria-hidden="true"><span /><span /><span /></div>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<Home />} />
          <Route path="/review" element={<Workstation />} />
          <Route path="/review/:id" element={<Workstation />} />
          <Route path="/workstation/*" element={<Navigate to="/review" replace />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/how-it-works" element={<Method />} />
        </Route>
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </SettingsContext.Provider>
  );
}
