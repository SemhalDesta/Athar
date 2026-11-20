import { useEffect, useMemo, useRef, useState } from "react";
import { Assessment, EvidenceRef, fmtDate, PatientRecord } from "../api";
import { laneName, RELEVANT } from "../insight";

interface Props {
  record: PatientRecord;
  a: Assessment;
  highlight: Set<string> | null;
  picked: string | null;
  onPick: (encounterId: string, ref?: EvidenceRef) => void;
}

/** The treatment ladder NICE NG73 describes for suspected endometriosis, bottom to top. */
const LEVELS = ["No treatment yet", "Painkillers", "Hormonal treatment", "Gynaecology referral"];
const levelOf = (r: EvidenceRef): number | null => {
  if (r.status === "referral") return /gyn/i.test(r.concept) ? 3 : null;
  if (r.status !== "medication") return null;
  if (/hormonal|progestogen/i.test(r.concept)) return 2;
  if (/nsaid|opioid|analgesic|antispasmodic/i.test(r.concept)) return 1;
  return null; // e.g. antibiotics do not move her up the ladder
};
const DAY = 86400000;
const monthYear = (d: string) => new Date(d + "T00:00:00").toLocaleDateString("en-GB", { month: "short", year: "numeric" });
const capWord = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export function journeySentence(a: Assessment): string {
  const painVisits = new Set(a.timeline.filter((t) => t.status === "present" && RELEVANT.test(t.concept)).map((t) => t.encounter_id)).size;
  if (!painVisits) return "No relevant pain is recorded in this patient's notes.";
  const tx = a.treatments_and_investigations;
  const tried: string[] = [];
  if (tx.some((r) => levelOf(r) === 1)) tried.push("painkillers");
  if (tx.some((r) => levelOf(r) === 2)) tried.push("hormonal treatment");
  const gyn = tx.find((r) => levelOf(r) === 3 && r.encounter_date);
  const persists = (a.features.persist_after_treatment ?? 0) > 0 || (a.features.documented_treatment_failure ?? 0) > 0;
  const normal = (a.features.persist_after_normal_investigation ?? 0) > 0;
  let s = `Pain was recorded at ${painVisits} visit${painVisits > 1 ? "s" : ""}.`;
  if (tried.length && persists) s += ` It continued despite ${tried.join(" and ")}${normal ? ", and after a normal test" : ""}.`;
  else if (tried.length) s += ` She was given ${tried.join(" and ")}.`;
  else s += " No pain treatment is recorded.";
  s += gyn ? ` She was referred to gynaecology in ${monthYear(gyn.encounter_date!)}.` : " She has not been referred to gynaecology.";
  return s;
}

/** Pain visits placed on the treatment step she was on at the time: does the pain keep coming back as care escalates? */
export default function PatternMap({ record, a, highlight, picked, onPick }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(900);
  const [tip, setTip] = useState<{ x: number; y: number; title: string; lines: string[] } | null>(null);
  useEffect(() => {
    const el = box.current!;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el); setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const data = useMemo(() => {
    const encs = record.encounters.filter((e) => e.date).sort((p, q) => (p.date! < q.date! ? -1 : 1));
    const ts = encs.map((e) => +new Date(e.date!));
    const t0 = (ts[0] ?? Date.now()) - 30 * DAY, t1 = (ts[ts.length - 1] ?? Date.now()) + 30 * DAY;
    const steps = a.treatments_and_investigations
      .map((r) => ({ r, lvl: levelOf(r) }))
      .filter((s): s is { r: EvidenceRef; lvl: number } => s.lvl !== null && !!s.r.encounter_date)
      .sort((p, q) => (p.r.encounter_date! < q.r.encounter_date! ? -1 : 1));
    const ups: { date: string; lvl: number; r: EvidenceRef }[] = [];
    let cur = 0;
    for (const s of steps) if (s.lvl > cur) { cur = s.lvl; ups.push({ date: s.r.encounter_date!, lvl: s.lvl, r: s.r }); }
    const levelBefore = (d: string) => ups.filter((u) => u.date < d).reduce((m, u) => Math.max(m, u.lvl), 0);
    const byEnc = new Map<string, EvidenceRef[]>();
    a.timeline.filter((t) => t.status === "present" && RELEVANT.test(t.concept) && t.encounter_date).forEach((t) => {
      byEnc.set(t.encounter_id, [...(byEnc.get(t.encounter_id) ?? []), t]);
    });
    const pain = [...byEnc.entries()].map(([enc, refs]) => ({
      enc, refs, date: refs[0].encounter_date!, lvl: levelBefore(refs[0].encounter_date!),
      types: [...new Set(refs.map((r) => laneName(r.concept)))],
      cyc: refs.some((r) => r.menstrual_link === "explicit_yes" || /painful periods/i.test(r.concept)),
    })).sort((p, q) => (p.date < q.date ? -1 : 1));
    const fails = a.treatments_and_investigations.filter((r) => r.status === "treatment_response" && /not working/i.test(r.concept) && r.encounter_date);
    const tests = a.treatments_and_investigations.filter((r) => r.status === "investigation" && r.encounter_date);
    const painEnc = new Set(pain.map((p) => p.enc));
    return { encs, t0, t1, ups, pain, fails, tests, painEnc, top: Math.max(1, cur, ...pain.map((p) => p.lvl)) };
  }, [record, a]);

  const LEFT = 168, RIGHT = 40, TOP = 66, ROW = 76;
  const plotW = Math.max(240, W - LEFT - RIGHT);
  const nLevels = LEVELS.length;
  const x = (d: string) => LEFT + ((+new Date(d) - data.t0) / (data.t1 - data.t0)) * plotW;
  const y = (lvl: number) => TOP + ROW * (nLevels - 1 - lvl) + 14;
  const baseY = y(0);
  const axisY = baseY + 58;
  const H = axisY + 40;

  let path = `M${LEFT},${y(0)}`;
  data.ups.forEach((u) => { path += ` H${x(u.date)} V${y(u.lvl)}`; });
  path += ` H${LEFT + plotW}`;
  const area = `${path} V${baseY + 18} H${LEFT} Z`;

  const placed: { x: number; y: number }[] = [];
  const dots = data.pain.map((p) => {
    const px = x(p.date); let py = y(p.lvl);
    while (placed.some((q) => Math.abs(q.x - px) < 20 && Math.abs(q.y - py) < 20)) py -= 22;
    placed.push({ x: px, y: py });
    return { ...p, px, py };
  });

  const segments = [0, ...data.ups.map((u) => u.lvl)].map((lvl, i) => {
    const from = i === 0 ? null : data.ups[i - 1].date, to = data.ups[i]?.date ?? null;
    const n = data.pain.filter((p) => (!from || p.date >= from) && (!to || p.date < to)).length;
    return { lvl, n, x0: from ? x(from) : LEFT, x1: to ? x(to) : LEFT + plotW };
  });

  const spanM = (data.t1 - data.t0) / (30 * DAY);
  const ticks: { d: string; label: string; major: boolean }[] = [];
  const c = new Date(data.t0); c.setDate(1); c.setMonth(c.getMonth() + 1);
  while (+c < data.t1) {
    const mo = c.getMonth();
    if (mo === 0 || (spanM <= 30 && mo % 3 === 0)) ticks.push({ d: c.toISOString().slice(0, 10), label: mo === 0 ? String(c.getFullYear()) : c.toLocaleDateString("en-GB", { month: "short" }), major: mo === 0 });
    c.setMonth(mo + 1);
  }

  const dim = (enc: string) => (highlight && !highlight.has(enc) ? " is-dim" : "");
  const show = (e: React.MouseEvent, title: string, lines: string[]) => {
    const r = box.current!.getBoundingClientRect();
    setTip({ x: e.clientX - r.left, y: e.clientY - r.top, title, lines });
  };

  return (
    <div className="journey" ref={box} onMouseLeave={() => setTip(null)}>
      <svg width={W} height={H} role="img" aria-label={journeySentence(a)}>
        <defs>
          <linearGradient id="jr-area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#b76cc8" stopOpacity="0.28" /><stop offset="1" stopColor="#b76cc8" stopOpacity="0.02" /></linearGradient>
        </defs>

        {LEVELS.map((l, i) => (
          <g key={l}>
            <line x1={LEFT} x2={LEFT + plotW} y1={y(i)} y2={y(i)} className="jr-row" />
            <text x={LEFT - 14} y={y(i) + 4} textAnchor="end" className={`jr-level ${i <= data.top ? "" : "faint"}`}>{l}</text>
          </g>
        ))}

        {data.tests.map((t, i) => {
          const tx = x(t.encounter_date!);
          const normal = /normal/.test(t.concept) && !/abnormal/.test(t.concept);
          const after = normal ? data.pain.filter((p) => p.date > t.encounter_date!).length : 0;
          const result = normal ? "normal" : (t.concept.match(/\((.*)\)/)?.[1] ?? "result");
          const name = t.concept.replace(/\s*\(.*\)/, "").replace("Pelvic ultrasound", "Ultrasound").replace("Full blood count / ferritin", "Blood count").replace("Urine culture / MSU", "Urine test");
          const ly = TOP - 30 + (i % 3) * 13;
          const right = tx > LEFT + plotW - 170;
          return (
            <g key={i} className={`jr-test${dim(t.encounter_id)}`} onClick={() => onPick(t.encounter_id, t)}
              onMouseMove={(e) => show(e, `${fmtDate(t.encounter_date)} · ${t.concept}`, [t.quote, ...(after ? [`Pain recorded at ${after} visit${after > 1 ? "s" : ""} after this normal result`] : [])])}>
              <line x1={tx} x2={tx} y1={ly + 4} y2={axisY} />
              <text x={right ? tx - 6 : tx + 6} y={ly} textAnchor={right ? "end" : "start"} className="jr-test-label">{name}: {result}</text>
            </g>
          );
        })}

        <path d={area} fill="url(#jr-area)" />
        <path d={path} className="jr-step" pathLength={1} />

        {segments.map((sg, i) => {
          const u = i > 0 ? data.ups[i - 1] : null;
          const what = u ? (u.r.status === "referral" ? "Referred" : `${capWord(u.r.quote)} started`) : "";
          const count = sg.n ? `${sg.n} pain visit${sg.n > 1 ? "s" : ""}${u ? " at this step" : " before any treatment"}` : "";
          const room = sg.x1 - sg.x0;
          const both = room > 230 && what && count;
          const text = both ? "both" : room > 70 ? (what || count) : "";
          return (
            <g key={i}>
              {u && <circle cx={sg.x0} cy={y(sg.lvl)} r={5} className={`jr-upc${dim(u.r.encounter_id)}`} onClick={() => onPick(u.r.encounter_id, u.r)}
                onMouseMove={(e) => show(e, `${fmtDate(u.date)} · ${LEVELS[u.lvl]}`, [u.r.status === "referral" ? u.r.quote : `${capWord(u.r.quote)} (${u.r.concept})`, ...(count ? [count] : [])])} />}
              {text && <text x={sg.x0 + 10} y={y(sg.lvl) + 34} className="jr-seg">{what && <tspan className="jr-seg-what">{what}</tspan>}{both ? ` · ${count}` : !what ? count : ""}</text>}
            </g>
          );
        })}

        {data.fails.map((f, i) => {
          const fx = x(f.encounter_date!), fy = axisY - 15;
          return (
            <g key={i} className={`jr-fail${dim(f.encounter_id)}`} onClick={() => onPick(f.encounter_id, f)} onMouseMove={(e) => show(e, `${fmtDate(f.encounter_date)} · Treatment not working`, [f.quote])}>
              <circle cx={fx} cy={fy} r={7} />
              <path d={`M${fx - 2.5},${fy - 2.5} l5,5 M${fx + 2.5},${fy - 2.5} l-5,5`} />
            </g>
          );
        })}

        {dots.map((p, i) => (
          <g key={p.enc} className={`jr-pain${dim(p.enc)}${picked === p.enc ? " is-picked" : ""}`} style={{ animationDelay: `${0.6 + i * 0.08}s` }}
            onClick={() => onPick(p.enc, p.refs[0])} onMouseMove={(e) => show(e, `${fmtDate(p.date)} · pain visit`, [p.types.join(", "), `“${p.refs[0].quote}”`])}>
            {p.cyc && <circle cx={p.px} cy={p.py} r={8 + p.types.length * 2 + 5} className="jr-cyc" />}
            <circle cx={p.px} cy={p.py} r={8 + p.types.length * 2} className="jr-dot" />
            {p.types.length > 1 && <text x={p.px} y={p.py + 4} textAnchor="middle" className="jr-n">{p.types.length}</text>}
          </g>
        ))}

        <line x1={LEFT} x2={LEFT + plotW} y1={axisY} y2={axisY} className="jr-axis" />
        {data.encs.map((e) => (
          <circle key={e.encounter_id} cx={x(e.date!)} cy={axisY} r={data.painEnc.has(e.encounter_id) ? 4.5 : 3} className={`jr-visit ${data.painEnc.has(e.encounter_id) ? "pain" : ""}${dim(e.encounter_id)}`}
            onClick={() => onPick(e.encounter_id)} onMouseMove={(ev) => show(ev, `${fmtDate(e.date)} · ${e.setting.replaceAll("_", " ")}`, [e.note_text.slice(0, 110) + (e.note_text.length > 110 ? "…" : "")])} />
        ))}
        <text x={LEFT - 14} y={axisY + 4} textAnchor="end" className="jr-level faint">Every visit</text>
        {ticks.map((t) => (
          <g key={t.d}>
            <line x1={x(t.d)} x2={x(t.d)} y1={axisY + 6} y2={axisY + (t.major ? 12 : 9)} className="jr-axis" />
            <text x={x(t.d)} y={axisY + 26} textAnchor="middle" className={`jr-tick ${t.major ? "major" : ""}`}>{t.label}</text>
          </g>
        ))}
      </svg>
      {tip && (
        <div className="pm-tip" style={{ left: Math.min(tip.x + 14, W - 290), top: tip.y + 14 }}>
          <b>{tip.title}</b>{tip.lines.map((l, i) => <span key={i}>{l}</span>)}
        </div>
      )}
    </div>
  );
}
