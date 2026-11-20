/** Small line icons shared by the home animation and patient review. */
export type Kind = "spark" | "pill" | "calendar" | "cycle" | "wave" | "note" | "flask" | "arrow" | "drop" | "cross" | "check";

export const ICON: Record<Kind, JSX.Element> = {
  spark: <path d="M13 2 5 14h6l-1 8 8-12h-6z" />,
  pill: <g transform="rotate(-38 12 12)"><rect x="3" y="8.5" width="18" height="7" rx="3.5" /><path d="M12 8.5v7" /></g>,
  calendar: <g><rect x="4" y="5" width="16" height="15" rx="2.5" /><path d="M4 10h16M8.5 3v4M15.5 3v4" /><circle cx="12" cy="15" r="1.3" /></g>,
  cycle: <g><path d="M19.5 12a7.5 7.5 0 1 1-2.6-5.7" /><path d="M19.5 4v4.5H15" /></g>,
  wave: <path d="M2.5 12h3.5l2-5 3.2 10 3-8 1.8 3h5.5" />,
  note: <g><rect x="5.5" y="3" width="13" height="18" rx="2.5" /><path d="M9 8h6M9 12h6M9 16h3.5" /></g>,
  flask: <path d="M9.5 3h5M10.5 3v6l-5 8.8A2 2 0 0 0 7.2 21h9.6a2 2 0 0 0 1.7-3.2l-5-8.8V3" />,
  arrow: <path d="M4 12h14M13 6.5l5.5 5.5-5.5 5.5" />,
  cross: <g><circle cx="12" cy="12" r="8.5" /><path d="M9 9l6 6M15 9l-6 6" /></g>,
  check: <g><circle cx="12" cy="12" r="8.5" /><path d="M8 12.5l2.7 2.7L16 9.5" /></g>,
  drop: <path d="M12 3.5c3 4 6 7.3 6 10.7a6 6 0 0 1-12 0c0-3.4 3-6.7 6-10.7z" />,
};

