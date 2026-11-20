import { Domain } from "../insight";

const STATE_TEXT = { documented: "Documented", uncertain: "Uncertain", denied: "Explicitly denied", unknown: "Never asked or recorded" };

function arc(cx: number, cy: number, r0: number, r1: number, a0: number, a1: number) {
  const p = (r: number, a: number) => `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
  const large = a1 - a0 > Math.PI ? 1 : 0;
  return `M${p(r1, a0)} A${r1},${r1} 0 ${large} 1 ${p(r1, a1)} L${p(r0, a1)} A${r0},${r0} 0 ${large} 0 ${p(r0, a0)} Z`;
}

/** Six key symptom groups. Unknown is drawn differently from denied: missing is not absent. */
export default function SymptomWheel({ domains, onPick, active }: { domains: Domain[]; onPick: (d: Domain) => void; active: string | null }) {
  const S = 236, cx = S / 2, cy = S / 2, r0 = 44, r1 = 82, gap = 0.05;
  const n = domains.length;
  const documented = domains.filter((d) => d.state === "documented").length;
  return (
    <figure className="wheel">
      <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label={`Symptom groups: ${domains.map((d) => `${d.label} ${STATE_TEXT[d.state].toLowerCase()}`).join("; ")}`}>
        <defs>
          <pattern id="wh-unc" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#f6ebf8" /><line x1="0" y1="0" x2="0" y2="6" stroke="#c99ad6" strokeWidth="2.2" /></pattern>
        </defs>
        {domains.map((d, i) => {
          const a0 = -Math.PI / 2 + (i / n) * Math.PI * 2 + gap, a1 = -Math.PI / 2 + ((i + 1) / n) * Math.PI * 2 - gap;
          const mid = (a0 + a1) / 2;
          const lx = cx + (r1 + 18) * Math.cos(mid), ly = cy + (r1 + 18) * Math.sin(mid);
          const op = d.state === "documented" ? 0.45 + 0.55 * Math.min(1, d.visits / 4) : 1;
          return (
            <g key={d.key} className={`wh-seg st-${d.state} ${active === d.key ? "is-active" : ""}`} onClick={() => onPick(d)} style={{ ["--op" as any]: op }}>
              <title>{`${d.label}: ${STATE_TEXT[d.state]}${d.visits ? `, ${d.visits} visit${d.visits > 1 ? "s" : ""}` : ""}`}</title>
              <path d={arc(cx, cy, r0, r1, a0, a1)} />
              {d.state === "documented" && <text x={cx + ((r0 + r1) / 2) * Math.cos(mid)} y={cy + ((r0 + r1) / 2) * Math.sin(mid) + 4} textAnchor="middle" className="wh-num">{d.visits}</text>}
              {d.state === "denied" && <text x={cx + ((r0 + r1) / 2) * Math.cos(mid)} y={cy + ((r0 + r1) / 2) * Math.sin(mid) + 5} textAnchor="middle" className="wh-num dn">–</text>}
              <text x={lx} y={ly + 4} textAnchor={Math.abs(Math.cos(mid)) < 0.3 ? "middle" : Math.cos(mid) > 0 ? "start" : "end"} className="wh-lab">{d.short}</text>
            </g>
          );
        })}
        <text x={cx} y={cy - 2} textAnchor="middle" className="wh-big">{documented}<tspan className="wh-of"> / {n}</tspan></text>
        <text x={cx} y={cy + 15} textAnchor="middle" className="wh-sub">documented</text>
      </svg>
      <figcaption className="wh-legend">
        <span><i className="lg doc" />Documented</span><span><i className="lg unc" />Uncertain</span><span><i className="lg den" />Denied</span><span><i className="lg unk" />Not recorded</span>
      </figcaption>
    </figure>
  );
}
