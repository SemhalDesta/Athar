import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { ICON, Kind } from "./icons";

const pct = (x: number | null | undefined) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);

interface Stage { key: string; who: string; role: string; icon: Kind; does: string; guard: string; detail: (r: any) => string[] }

const STAGES: Stage[] = [
  {
    key: "record", who: "Patient record", role: "Input", icon: "note",
    does: "Years of routine visits: free-text notes, coded diagnoses, prescriptions, test results, referrals, age and family history.",
    guard: "No vital signs are used. Endometriosis does not show up in blood pressure, heart rate or temperature, and NICE advises against using CA-125 or MRI to diagnose it. The pattern lives in the history.",
    detail: () => ["Any record in Athar's patient format can be assessed, so real EHR exports can replace the synthetic data."],
  },
  {
    key: "reader", who: "LLM reader", role: "AI", icon: "spark",
    does: "Reads every note and returns each symptom as present, denied or uncertain, with its link to periods, severity, effect on daily life and response to treatment.",
    guard: "Every finding must come with the exact sentence it relied on. Code searches the note for that sentence and discards the finding if it is not there. A symptom nobody wrote about is recorded as unknown, never as absent.",
    detail: (r) => {
      const c = r?.extractor_comparison;
      const kw = c && Object.entries(c).find(([k]) => k.startsWith("lexicon"))?.[1] as any;
      const llm = c && Object.entries(c).find(([k]) => k.startsWith("llm"))?.[1] as any;
      return kw && llm && llm.symptom_recall ? [`Found ${pct(llm.symptom_recall)} of the symptoms written in the test notes, against ${pct(kw.symptom_recall)} for keyword rules.`] : [];
    },
  },
  {
    key: "timeline", who: "Timeline", role: "Code", icon: "calendar",
    does: "Places every finding on the patient's timeline and measures the pattern: how often pain recurs and over how long, whether it follows her cycle, and whether it continues after treatment or after a normal test.",
    guard: "This step is plain code, so the same record always gives the same measurements. Copied notes count once, and impossible dates are excluded from timing.",
    detail: () => ["Patterns of symptoms, prescribing and tests over time appeared years before diagnosis in GP records in a Scottish primary-care study (Burton 2017)."],
  },
  {
    key: "model", who: "Model", role: "ML", icon: "wave",
    does: "Scores how closely the documented pattern resembles that of patients who had endometriosis. SHAP shows how much each measurement pushed the score up or down.",
    guard: "The score is not a probability and is never shown as one. Three model types were compared on the development patients only, and the threshold was set to catch at least 80% of cases there.",
    detail: (r) => {
      const m = r?.model;
      return m ? [`Chosen: ${String(m.chosen_model).replaceAll("_", " ")}, review threshold ${m.threshold}.`] : [];
    },
  },
  {
    key: "rules", who: "Clinical rules", role: "Code", icon: "check",
    does: "Turn the score into one of three outcomes: the pattern warrants clinical review, insufficient information, or no notable pattern identified.",
    guard: "Too little history always gives 'insufficient information', whatever the score. A single visit, nonspecific symptoms alone, or another diagnosis that explains every symptom can block a flag, but no rule can create one. A normal scan never lowers suspicion.",
    detail: () => ["All rule settings live in one configuration file, each with the evidence that motivated it."],
  },
  {
    key: "sceptic", who: "Sceptic reviewer", role: "AI", icon: "cross",
    does: "A second LLM call argues against every flag. It looks for misread notes, other documented explanations and gaps in the record.",
    guard: "Every challenge must quote the note, and code discards any quote it cannot find. A verified misread is corrected and the record re-scored; a confirmed alternative downgrades the flag. The sceptic can only hold a flag back, never create one.",
    detail: (r) => {
      const v = r?.sceptic?.by_verdict;
      if (!v || !(v.upheld?.n)) return [];
      const wn = (v.weakened?.n ?? 0) + (v.challenged?.n ?? 0), wp = (v.weakened?.label_positive ?? 0) + (v.challenged?.label_positive ?? 0);
      return [`In testing, ${pct(v.upheld.label_positive / v.upheld.n)} of flags it upheld were correct, against ${pct(wn ? wp / wn : 0)} of flags it weakened, so upheld flags are shown first.`];
    },
  },
  {
    key: "doctor", who: "Doctor", role: "Human", icon: "arrow",
    does: "Reviews the pattern, the evidence behind each point and the sceptic's challenges, then agrees to assess, asks for more information, or dismisses the flag.",
    guard: "Athar never diagnoses. Dismissing a flag requires a reason, and every decision is recorded with the evidence the doctor saw.",
    detail: () => [],
  },
];

/** Which group each model feature belongs to; the list itself comes from the trained model. */
const GROUP_OF: Record<string, string> = {
  n_dysmenorrhoea: "Symptoms, counted by visit", n_pelvic_pain: "Symptoms, counted by visit", n_dyspareunia: "Symptoms, counted by visit",
  n_dyschezia: "Symptoms, counted by visit", n_urinary: "Symptoms, counted by visit", n_subfertility: "Symptoms, counted by visit",
  n_hmb: "Symptoms, counted by visit", n_weak_symptoms: "Symptoms, counted by visit", n_strong_domains: "Symptoms, counted by visit", n_domains: "Symptoms, counted by visit",
  n_relevant_encounters: "Timing", pain_span_months: "Timing", escalation: "Timing", record_span_months: "Timing",
  n_cyclical_nonmenstrual: "Link to the cycle", n_explicit_noncyclical: "Link to the cycle",
  severe_or_impact: "Severity",
  persist_after_treatment: "Response to treatment", documented_treatment_failure: "Response to treatment", documented_treatment_success: "Response to treatment",
  persist_after_normal_investigation: "Tests", suggestive_imaging: "Tests",
  n_alt_explanations: "Other explanations", alt_explains_all: "Other explanations",
  gyn_referral: "Context", family_history: "Context", age: "Context", n_uncertain: "Context", documentation_completeness: "Context",
};
const GROUP_ORDER = ["Symptoms, counted by visit", "Timing", "Link to the cycle", "Severity", "Response to treatment", "Tests", "Other explanations", "Context"];

const REFS: { cite: string; url: string; use: string }[] = [
  { cite: "NICE. Endometriosis: diagnosis and management (NG73). 2017, updated November 2024.", url: "https://www.nice.org.uk/guidance/ng73", use: "Symptoms that should raise suspicion, initial management and referral, and that a normal ultrasound does not rule it out." },
  { cite: "Becker CM et al. ESHRE guideline: endometriosis. Human Reproduction Open 2022; hoac009.", url: "https://pubmed.ncbi.nlm.nih.gov/35350465/", use: "Clinical presentation and the diagnostic pathway." },
  { cite: "Burton C et al. Pointers to earlier diagnosis of endometriosis: a nested case-control study using primary care electronic health records. British Journal of General Practice 2017; 67(665): e816.", url: "https://bjgp.org/content/67/665/e816", use: "Patterns of symptoms, prescribing and tests over time as pointers in GP records." },
  { cite: "Mazza D et al. The clinical presentation, investigation, and management of women diagnosed with endometriosis in Australian general practices, 2011–2021. Medical Journal of Australia 2025; 223(11).", url: "https://www.mja.com.au/journal/2025/223/11/clinical-presentation-investigation-and-management-women-diagnosed", use: "How varied the documented symptoms are before diagnosis, and the time from first symptoms to diagnosis." },
  { cite: "Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. NeurIPS 2017.", url: "https://arxiv.org/abs/1705.07874", use: "SHAP values used to explain each score." },
  { cite: "Chapman WW et al. A simple algorithm for identifying negated findings and diseases in discharge summaries. Journal of Biomedical Informatics 2001; 34(5): 301–310.", url: "https://pubmed.ncbi.nlm.nih.gov/12123149/", use: "Negation handling in the keyword baseline reader." },
];

export default function Method() {
  const [r, setR] = useState<any>(null);
  const [model, setModel] = useState<any>(null);
  const [sel, setSel] = useState("reader");
  useEffect(() => { api.evaluation().then(setR).catch(() => {}); api.model().then(setModel).catch(() => {}); }, []);
  const feats: string[] = model?.features ?? [];
  const groups = GROUP_ORDER.map((g) => [g, feats.filter((f) => (GROUP_OF[f] ?? "Context") === g)] as [string, string[]]).filter(([, fs]) => fs.length);
  const st = STAGES.find((s) => s.key === sel)!;
  const extra = st.detail(r);

  return (
    <div className="hiw">
      <header className="perf-head">
        <h1>How Athar works</h1>
        <p className="lede">Endometriosis is often missed because its signs are spread across many short visits. Athar reads the whole history, measures the pattern guidelines describe, and shows the doctor every step it took. Select a step to see what it does and what keeps it honest.</p>
      </header>

      <section className="card pipe-card">
        <ol className="pipe" aria-label="Athar pipeline">
          {STAGES.map((s, i) => (
            <li key={s.key} style={{ ["--i" as any]: i }}>
              <button className={`pipe-node r-${s.role.toLowerCase()} ${sel === s.key ? "is-on" : ""}`} onClick={() => setSel(s.key)} aria-pressed={sel === s.key}>
                <span className="pipe-icon"><svg viewBox="0 0 24 24" aria-hidden>{ICON[s.icon]}</svg></span>
                <span className="pipe-who">{s.who}</span>
                <span className="pipe-role">{s.role}</span>
              </button>
              {i < STAGES.length - 1 && <span className="pipe-link" aria-hidden><i /></span>}
            </li>
          ))}
        </ol>
        <div className="pipe-detail" key={st.key}>
          <div><h2>{st.who}</h2><p className="pd-role">{st.role === "AI" ? "An AI step" : st.role === "ML" ? "A machine learning step" : st.role === "Code" ? "A deterministic code step" : st.role === "Human" ? "The human decision" : "What goes in"}</p></div>
          <div><h3>What it does</h3><p>{st.does}</p></div>
          <div><h3>What keeps it honest</h3><p>{st.guard}</p>{extra.map((x, i) => <p key={i} className="pd-extra">{x}</p>)}</div>
        </div>
      </section>

      <section className="card perf-block">
        <h2>The {feats.length || 29} measurements the model uses</h2>
        <p>This list is read from the trained model itself, so it always matches what the model scores. Every measurement is computed by code from the extracted findings, and in patient review each one links back to the visits and sentences it came from.</p>
        {feats.length ? (
          <div className="feat-groups">
            {groups.map(([g, fs]) => (
              <div key={g} className="fg">
                <h3>{g}</h3>
                <ul>{fs.map((f) => <li key={f}><span>{model.feature_labels?.[f] ?? f}</span><code>{f}</code></li>)}</ul>
              </div>
            ))}
          </div>
        ) : <p className="muted small">Start the backend to load the model's feature list.</p>}
      </section>

      <section className="card perf-block two">
        <div>
          <h2>How the test patients were made</h2>
          <ul className="steps-list">
            <li>Each patient's hidden label (endometriosis or not) was set first.</li>
            <li>A scenario was then chosen to fit the label, such as a textbook pattern, pain labelled as IBS, a thin record, or routine visits only. Some positive patients have little relevant in their notes, as in real life.</li>
            <li>A structured history was generated and written up as visit notes, then reworded by an LLM so the language varies.</li>
            <li>400 development patients were used to build and train. 200 evaluation patients, generated separately with different wording, were held back for testing.</li>
            <li>The hidden labels never reach the pipeline. They are used only to score it.</li>
          </ul>
          <Link to="/performance" className="link-row" style={{ paddingLeft: 0 }}>See the results ›</Link>
        </div>
        <div>
          <h2>What it cannot do</h2>
          <ul className="steps-list">
            <li>It only sees what was written down. Pain that was never recorded cannot be found.</li>
            <li>Adenomyosis, IBS, bladder pain syndrome and pelvic inflammatory disease share symptoms, so some flags will be for other conditions. The sceptic and the doctor exist for this.</li>
            <li>It does not diagnose. It points to records that deserve a closer look.</li>
          </ul>
        </div>
      </section>

      <section className="card perf-block">
        <h2>References</h2>
        <ol className="refs">
          {REFS.map((x) => <li key={x.url}><a href={x.url} target="_blank" rel="noreferrer">{x.cite}</a><span>{x.use}</span></li>)}
        </ol>
      </section>
    </div>
  );
}
