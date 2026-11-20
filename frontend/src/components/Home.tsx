import { Link } from "react-router-dom";
import ClueField from "./ClueField";

export default function Home() {
  return (
    <div className="home">
      <section className="home-hero">
        <div className="hh-text">
          <h1><span>The clues were there.</span><span>The pattern was hidden.</span></h1>
          <p className="hh-sub">Endometriosis takes years to diagnose, yet the signs are often already in the record, scattered across short visits. Athar reads every note, prescription and test result, connects them over time, and shows the doctor exactly why a record deserves a closer look.</p>
          <div className="hh-cta">
            <Link to="/review" className="btn btn-primary">Open patient review</Link>
            <Link to="/performance" className="btn btn-ghost">See how it performs</Link>
          </div>
        </div>
        <ClueField />
      </section>

      <section className="home-band">
        <div className="hb-step">
          <h3>Scattered across visits</h3>
          <div className="snips">
            <div className="snip"><span>2021</span>c/o painful periods, ibuprofen advised</div>
            <div className="snip"><span>2022</span>LIF pain x3/7, ?IBS</div>
            <div className="snip"><span>2023</span>Pain persisting despite the pill</div>
            <div className="snip"><span>2024</span>TVUS: normal. Review PRN</div>
          </div>
          <p>Each note looks minor on its own, and no single visit raises concern.</p>
        </div>
        <div className="hb-step">
          <h3>Connected over time</h3>
          <p>AI reads the notes and quotes the exact sentence behind every finding. Athar then measures recurrence, the link to her cycle, and pain that continues after treatment or a normal scan.</p>
          <p>A model trained on these patterns scores the record, and every point of the score is explained.</p>
        </div>
        <div className="hb-step">
          <h3>Reviewed by her doctor</h3>
          <p>The doctor sees one of three outcomes: the pattern warrants review, there isn't enough information, or no notable pattern. Each comes with the reasons, the missing information, and other explanations in the record.</p>
          <p>Athar never diagnoses. The doctor decides.</p>
        </div>
      </section>

      <section className="home-creed">
        <div><h3>It shows its working.</h3><p>Every conclusion links to the visit and the sentence it came from.</p></div>
        <div><h3>It respects what is unknown.</h3><p>A symptom nobody asked about is unknown, not absent. Thin records are marked as needing more history.</p></div>
        <div><h3>It can only hold back.</h3><p>Clinical guardrails can block a flag, never create one. Its scores are not probabilities.</p></div>
      </section>

      <footer className="site-foot">Athar is a hackathon prototype for clinical decision support. It is not a medical device and does not diagnose endometriosis. All patients shown are fictional.</footer>
    </div>
  );
}
