import { createContext, useContext } from "react";

export interface Settings { dataset: string; showTruth: boolean }
export const SettingsContext = createContext<{ settings: Settings; setSettings: (s: Settings) => void }>({
  settings: { dataset: "eval", showTruth: false }, setSettings: () => {},
});
export const useSettings = () => useContext(SettingsContext);
