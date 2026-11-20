import { useEffect, useRef, useState } from "react";
import { BrandMark } from "./Brand";
import { ICON, Kind } from "./icons";

/**
 * Home-page centrepiece: small clue icons (prescriptions, cycle notes, scans, pain entries)
 * drift through a slowly rotating 3D field. Every cycle the clues that belong to one patient's
 * story are pulled into a tilted ring and threaded together, while unrelated clues fade back.
 */

/** The clues that make up one fictional patient's pattern, in time order. */
const STORY: { kind: Kind; label: string }[] = [
  { kind: "spark", label: "2021 · Painful periods" },
  { kind: "pill", label: "2021 · Ibuprofen advised" },
  { kind: "calendar", label: "2022 · Pain worse before periods" },
  { kind: "spark", label: "2022 · Pelvic pain, ?IBS" },
  { kind: "pill", label: "2023 · Combined pill started" },
  { kind: "cycle", label: "2023 · Pain despite the pill" },
  { kind: "wave", label: "2024 · Ultrasound normal" },
  { kind: "spark", label: "2024 · Pain after normal scan" },
  { kind: "note", label: "2025 · Missed work with pain" },
  { kind: "arrow", label: "2025 · Seen again, same pain" },
];
const NOISE: Kind[] = ["note", "flask", "drop", "pill", "note", "wave", "flask", "calendar", "arrow", "note", "drop", "flask",
  "note", "pill", "arrow", "calendar", "note", "flask", "drop", "wave", "note", "flask", "note", "arrow"];

interface Clue { kind: Kind; label?: string; story: number; ux: number; uy: number; uz: number; ph: number; sp: number }

function makeClues(): Clue[] {
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const all: Clue[] = [];
  const place = (kind: Kind, story: number, label?: string) =>
    all.push({ kind, label, story, ux: rnd() * 2 - 1, uy: rnd() * 2 - 1, uz: rnd() * 2 - 1, ph: rnd() * Math.PI * 2, sp: 0.5 + rnd() * 0.7 });
  STORY.forEach((s, i) => place(s.kind, i, s.label));
  NOISE.forEach((k) => place(k, -1));
  return all;
}

const CYCLE = 15; // seconds
const smooth = (x: number) => { const t = Math.min(1, Math.max(0, x)); return t * t * (3 - 2 * t); };
function gatherAt(t: number) {
  const c = t % CYCLE;
  if (c < 4.5) return 0;
  if (c < 7) return smooth((c - 4.5) / 2.5);
  if (c < 12) return 1;
  if (c < 14) return 1 - smooth((c - 12) / 2);
  return 0;
}

export default function ClueField() {
  const stage = useRef<HTMLDivElement>(null);
  const nodes = useRef<(HTMLDivElement | null)[]>([]);
  const labels = useRef<(HTMLSpanElement | null)[]>([]);
  const loop = useRef<SVGPolygonElement>(null);
  const spokes = useRef<SVGGElement>(null);
  const core = useRef<HTMLDivElement>(null);
  const clues = useRef<Clue[]>(makeClues());
  const [formed, setFormed] = useState(false);

  useEffect(() => {
    const el = stage.current!;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W = el.clientWidth, H = el.clientHeight;
    const ro = new ResizeObserver(() => { W = el.clientWidth; H = el.clientHeight; });
    ro.observe(el);
    let raf = 0, last = -1;
    const start = performance.now() - (reduced ? 9000 : 1500);

    const frame = (now: number) => {
      const t = reduced ? 9 : (now - start) / 1000;
      const k = gatherAt(t);
      const cx = W / 2, cy = H / 2;
      const spin = t * 0.07;                       // whole field turns slowly
      const ringSpin = t * 0.32;                   // the pattern ring turns faster
      const rr = Math.min(W * 0.36, H * 0.42);
      const pts: [number, number][] = [];
      const cs = clues.current;
      const n = STORY.length;

      cs.forEach((c, i) => {
        const node = nodes.current[i];
        if (!node) return;
        // drifting position in the unit cube, rotated about the vertical axis
        const dx = c.ux + 0.06 * Math.sin(t * 0.35 * c.sp + c.ph);
        const dy = c.uy + 0.07 * Math.cos(t * 0.3 * c.sp + c.ph * 1.3);
        const dz = c.uz;
        const rx = dx * Math.cos(spin) - dz * Math.sin(spin);
        const rz = dx * Math.sin(spin) + dz * Math.cos(spin);
        let x = cx + rx * W * 0.46, y = cy + dy * H * 0.42, depth = rz;

        if (c.story >= 0) {
          const th = (c.story / n) * Math.PI * 2 + ringSpin;
          const tx = cx + Math.cos(th) * rr, ty = cy + Math.sin(th) * rr * 0.34, td = Math.sin(th);
          x += (tx - x) * k; y += (ty - y) * k; depth += (td - depth) * k;
          pts[c.story] = [x, y];
        }
        const near = (depth + 1) / 2;              // 0 far … 1 near
        let scale = 0.62 + 0.5 * near, op = 0.35 + 0.65 * near;
        if (c.story < 0) { op *= 1 - 0.72 * k; scale *= 1 - 0.18 * k; }
        else { scale *= 1 + 0.22 * k; op = op + (1 - op) * k * 0.85; }
        node.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%) scale(${scale.toFixed(3)})`;
        node.style.opacity = op.toFixed(3);
        node.style.zIndex = String(Math.round(near * 100) + (c.story >= 0 ? 100 : 0));
        node.classList.toggle("is-story", c.story >= 0 && k > 0.5);

        const lab = c.story >= 0 ? labels.current[c.story] : null;
        if (lab) lab.style.opacity = (k > 0.95 ? smooth((depth - 0.86) / 0.12) : 0).toFixed(3);
      });

      if (loop.current && pts.length === n) {
        loop.current.setAttribute("points", pts.map((p) => p.join(",")).join(" "));
        loop.current.style.opacity = String(Math.pow(k, 3) * 0.75);
      }
      if (spokes.current) {
        Array.from(spokes.current.children).forEach((ln, i) => {
          const p = pts[i];
          if (!p) return;
          ln.setAttribute("x1", String(cx)); ln.setAttribute("y1", String(cy));
          ln.setAttribute("x2", String(p[0])); ln.setAttribute("y2", String(p[1]));
        });
        spokes.current.style.opacity = String(Math.pow(k, 3) * 0.35);
      }
      if (core.current) {
        core.current.style.transform = `translate(-50%, -50%) scale(${(0.55 + 0.45 * k).toFixed(3)})`;
        core.current.style.opacity = String(0.15 + 0.85 * k);
      }
      const f = k > 0.6;
      if (f !== (last === 1)) { last = f ? 1 : 0; setFormed(f); }
      if (!reduced) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return (
    <figure className="cluefield" aria-label="Animation: scattered clues from many visits come together into one pattern">
      <div className="cf-stage" ref={stage}>
        <div className="cf-glow" aria-hidden />
        <svg className="cf-threads" aria-hidden>
          <g ref={spokes}>{STORY.map((_, i) => <line key={i} />)}</g>
          <polygon ref={loop} />
        </svg>
        <div className="cf-core" ref={core} aria-hidden><BrandMark /></div>
        {clues.current.map((c, i) => (
          <div key={i} className={`cf-clue ${c.story >= 0 ? "story" : ""}`} ref={(n) => { nodes.current[i] = n; }} aria-hidden>
            <svg viewBox="0 0 24 24">{ICON[c.kind]}</svg>
            {c.story >= 0 && <span className="cf-label" ref={(n) => { labels.current[c.story] = n; }}>{c.label}</span>}
          </div>
        ))}
      </div>
      <figcaption className="cf-caption" aria-live="polite">
        <span className={formed ? "" : "on"}>34 entries across 6 years of visits</span>
        <span className={formed ? "on" : ""}>10 of them form one pattern <b className="pill pill-review">Warrants clinical review</b></span>
      </figcaption>
    </figure>
  );
}
