"""Clinical vocabulary shared by the generator and the lexicon extractor.

Tiers follow the Athar Clinical Evidence Document (NICE NG73, ESHRE 2022, Ballard 2008):
  strong     – guideline features with large effect sizes in routine records
  supporting – adds weight in combination, never alone
  weak       – common / nonspecific; can never trigger a flag
`gen_rare` phrasings are used by the generator but deliberately NOT all covered by
lexicon patterns, so the keyword baseline misses some wording that an LLM should catch.
"""
from __future__ import annotations

SYMPTOMS: dict[str, dict] = {
    "dysmenorrhoea": {
        "label": "Painful periods (dysmenorrhoea)",
        "tier": "strong",
        "patterns": [
            r"dysmenorr?h?o?ea", r"dysmen\b", r"period pains?", r"painful periods?",
            r"menstrual (?:pain|cramps?)", r"painful menses", r"pain (?:with|during) (?:her )?(?:periods?|menses)",
            r"crampy pain with menses", r"period cramps?",
        ],
        "gen": [
            "dysmenorrhoea", "painful periods", "period pain", "severe menstrual cramps",
            "pain with her periods", "dysmen", "painful menses",
        ],
        "gen_rare": ["crippling cramps each month", "monthly pain when bleeding", "terrible cramps on day 1-2"],
    },
    "pelvic_pain": {
        "label": "Pelvic / lower abdominal pain",
        "tier": "strong",
        "patterns": [
            r"pelvic pain", r"lower abdo(?:minal)? pain", r"\bLIF pain", r"\bRIF pain", r"suprapubic pain",
            r"lower abdominal discomfort", r"pain in (?:the )?pelvis", r"\bCPP\b", r"adnexal (?:pain|tenderness)",
            r"(?:left|right) iliac fossa pain",
        ],
        "gen": [
            "pelvic pain", "lower abdominal pain", "LIF pain", "RIF pain", "suprapubic pain",
            "lower abdo pain", "pain in the pelvis",
        ],
        "gen_rare": ["dragging ache low down", "deep ache in the lower belly", "pain below the umbilicus"],
    },
    "dyspareunia": {
        "label": "Deep pain during/after sex (dyspareunia)",
        "tier": "strong",
        "patterns": [
            r"dyspareunia", r"pain (?:during|with|after) (?:sex|intercourse)", r"painful (?:sex|intercourse)",
            r"deep pain (?:on|with) penetration", r"post-?coital pain",
        ],
        "gen": ["deep dyspareunia", "pain during intercourse", "painful sex", "pain after intercourse"],
        "gen_rare": ["intimacy is painful deep inside", "pain with relations"],
    },
    "dyschezia": {
        "label": "Painful bowel movements (dyschezia)",
        "tier": "strong",
        "patterns": [
            r"dyschezia", r"pain(?:ful)? (?:on|with|when) (?:opening (?:her )?bowels|defecat\w+|passing stool)",
            r"painful bowel movements?", r"pain on defecation", r"painful defecation", r"\bpain opening bowels",
        ],
        "gen": ["dyschezia", "pain on opening bowels", "painful bowel movements", "pain on defecation"],
        "gen_rare": ["it hurts to go to the toilet (bowels)", "stabbing pain passing stool"],
    },
    "urinary": {
        "label": "Urinary pain / blood in urine",
        "tier": "supporting",
        "patterns": [r"dysuria", r"haematuria", r"hematuria", r"pain (?:on |when )?passing urine", r"burning (?:on|when) (?:passing urine|urination)", r"blood in (?:the |her )?urine"],
        "gen": ["dysuria", "pain passing urine", "haematuria", "burning on urination"],
        "gen_rare": ["stinging when she pees"],
    },
    "heavy_menstrual_bleeding": {
        "label": "Heavy menstrual bleeding",
        "tier": "weak",
        "patterns": [r"menorrhagia", r"heavy (?:menstrual )?(?:periods?|bleeding|menses)", r"\bHMB\b", r"flooding", r"passing clots"],
        "gen": ["heavy periods", "menorrhagia", "HMB", "heavy menstrual bleeding with clots"],
        "gen_rare": ["changing pads hourly"],
    },
    "subfertility": {
        "label": "Difficulty conceiving",
        "tier": "supporting",
        "patterns": [r"subfertility", r"infertility", r"trying to conceive", r"\bTTC\b", r"unable to conceive", r"not (?:yet )?(?:pregnant|conceived) after"],
        "gen": ["trying to conceive for 18 months", "subfertility", "TTC 2 years without success"],
        "gen_rare": ["keen for a baby, nothing yet after 2y"],
    },
    "fatigue": {
        "label": "Fatigue",
        "tier": "weak",
        "patterns": [r"fatigue", r"tired all the time", r"\bTATT\b", r"exhaust(?:ed|ion)", r"lethargy"],
        "gen": ["fatigue", "TATT", "tired all the time", "exhausted"],
        "gen_rare": [],
    },
    "bloating": {
        "label": "Bloating",
        "tier": "weak",
        "patterns": [r"bloat(?:ing|ed)", r"abdominal distension"],
        "gen": ["bloating", "abdominal bloating"],
        "gen_rare": [],
    },
    "bowel_change": {
        "label": "Altered bowel habit (diarrhoea/constipation)",
        "tier": "weak",
        "patterns": [r"diarrh?o?ea", r"constipation", r"altered bowel habit", r"loose stools"],
        "gen": ["altered bowel habit", "loose stools", "constipation", "diarrhoea"],
        "gen_rare": [],
    },
    "back_pain": {
        "label": "Back pain",
        "tier": "weak",
        "patterns": [r"(?:low(?:er)? )?back (?:pain|ache)", r"backache", r"lumbar pain"],
        "gen": ["low back pain", "backache"],
        "gen_rare": [],
    },
}

PAIN_CONCEPTS = {"dysmenorrhoea", "pelvic_pain", "dyspareunia", "dyschezia", "urinary"}
# Domains used for the "symptom domain count" feature.
DOMAINS = {
    "dysmenorrhoea": "menstrual_pain",
    "pelvic_pain": "pelvic_pain",
    "dyspareunia": "dyspareunia",
    "dyschezia": "bowel_pain",
    "urinary": "urinary",
    "subfertility": "fertility",
}

MEDICATIONS: dict[str, dict] = {
    "nsaid": {"label": "NSAID", "names": ["ibuprofen", "naproxen", "mefenamic acid", "diclofenac"], "treatment": True},
    "combined_hormonal": {"label": "Combined hormonal contraceptive", "names": ["Microgynon", "Rigevidon", "combined pill", "COCP", "Yasmin"], "treatment": True},
    "progestogen": {"label": "Progestogen-only treatment", "names": ["desogestrel", "POP", "norethisterone", "Mirena", "LNG-IUS", "Depo-Provera"], "treatment": True},
    "opioid": {"label": "Opioid analgesic", "names": ["codeine", "co-codamol", "tramadol"], "treatment": True},
    "tranexamic": {"label": "Tranexamic acid", "names": ["tranexamic acid"], "treatment": False},
    "antispasmodic": {"label": "Antispasmodic", "names": ["mebeverine", "buscopan", "hyoscine"], "treatment": False},
    "antibiotic": {"label": "Antibiotic", "names": ["nitrofurantoin", "trimethoprim", "doxycycline", "metronidazole", "ceftriaxone"], "treatment": False},
    "neuropathic": {"label": "Neuropathic analgesic", "names": ["amitriptyline", "gabapentin", "pregabalin"], "treatment": True},
}
# Medication classes that count as "treatment for pain" when assessing persistence.
PAIN_TREATMENT_CLASSES = {k for k, v in MEDICATIONS.items() if v["treatment"]}

INVESTIGATIONS = {
    "pelvic_ultrasound": {"label": "Pelvic ultrasound", "patterns": [r"\bTVUS\b", r"(?:pelvic|transvaginal) (?:ultrasound|USS?|scan)", r"\bUSS\b"]},
    "mri_pelvis": {"label": "MRI pelvis", "patterns": [r"\bMRI\b"]},
    "ca125": {"label": "CA125", "patterns": [r"CA-?125"]},
    "urine_culture": {"label": "Urine culture / MSU", "patterns": [r"\bMSU\b", r"urine (?:culture|dip)"]},
    "sti_screen": {"label": "STI screen", "patterns": [r"STI screen", r"chlamydia"]},
    "colonoscopy": {"label": "Colonoscopy", "patterns": [r"colonoscopy"]},
    "fbc": {"label": "Full blood count / ferritin", "patterns": [r"\bFBC\b", r"ferritin"]},
    "laparoscopy": {"label": "Diagnostic laparoscopy", "patterns": [r"laparoscopy"]},
}

# Imaging findings considered strong (endometrioma / deep nodule).
STRONG_IMAGING_TERMS = [r"endometrioma", r"chocolate cyst", r"deep (?:endometriotic )?nodule", r"kissing ovaries", r"obliterated pouch of douglas"]
NORMAL_RESULT_TERMS = [r"\bnormal\b", r"\bNAD\b", r"unremarkable", r"no abnormality", r"negative", r"within normal limits", r"\bWNL\b", r"no growth"]

ALTERNATIVE_DIAGNOSES = {
    "ibs": {"label": "Irritable bowel syndrome", "patterns": [r"\bIBS\b", r"irritable bowel"], "explains": {"bowel_change", "bloating", "pelvic_pain"}, "caution": "IBS is often a co-diagnosis or misdiagnosis in endometriosis (Seaman 2008); it does not exclude it."},
    "uti": {"label": "Urinary tract infection", "patterns": [r"\bUTI\b", r"urinary tract infection", r"cystitis"], "explains": {"urinary"}, "caution": ""},
    "pid": {"label": "Pelvic inflammatory disease", "patterns": [r"\bPID\b", r"pelvic inflammatory"], "explains": {"pelvic_pain", "dyspareunia"}, "caution": "PID treatment is more common before an endometriosis diagnosis (Seaman 2008)."},
    "fibroids": {"label": "Uterine fibroids", "patterns": [r"fibroids?", r"leiomyoma"], "explains": {"heavy_menstrual_bleeding", "pelvic_pain"}, "caution": ""},
    "adenomyosis": {"label": "Adenomyosis", "patterns": [r"adenomyosis"], "explains": {"dysmenorrhoea", "heavy_menstrual_bleeding"}, "caution": "Adenomyosis frequently coexists with endometriosis."},
    "simple_cyst": {"label": "Simple / functional ovarian cyst", "patterns": [r"simple (?:ovarian )?cyst", r"functional cyst", r"follicular cyst"], "explains": {"pelvic_pain"}, "caution": ""},
    "msk_back": {"label": "Musculoskeletal back pain", "patterns": [r"mechanical back pain", r"musculoskeletal"], "explains": {"back_pain"}, "caution": ""},
    "primary_dysmenorrhoea": {"label": "Primary dysmenorrhoea", "patterns": [r"primary dysmenorr?h?o?ea"], "explains": {"dysmenorrhoea"}, "caution": "A label of primary dysmenorrhoea is a working diagnosis; NICE advises suspicion when pain persists despite treatment."},
    "interstitial_cystitis": {"label": "Bladder pain syndrome", "patterns": [r"interstitial cystitis", r"bladder pain syndrome"], "explains": {"urinary", "pelvic_pain"}, "caution": ""},
}

# Text cues
NEGATION_CUES = [r"\bno\b", r"\bdenies\b", r"\bdenied\b", r"\bnot\b", r"\bnil\b", r"\bnegative for\b", r"\bwithout\b", r"\babsent\b", r"\bnone\b", r"\bfree of\b"]
UNCERTAINTY_CUES = [r"\?", r"\bpossible\b", r"\bpossibly\b", r"\bquery\b", r"\bunclear\b", r"\bmay be\b", r"\bunsure\b", r"\b(?:might|could) be\b", r"\bequivocal\b", r"\bnot sure\b", r"\bvague\b"]
MENSTRUAL_LINK_YES = [
    r"with (?:her )?(?:periods?|menses|cycle)", r"during (?:her )?(?:periods?|menses|menstruation)", r"cyclical",
    r"around (?:her |the time of (?:her )?)?(?:periods?|menses)", r"premenstrual", r"(?:worse|flares?|peaks?) (?:at|with|before|around) (?:her )?(?:periods?|menses|menstruation)",
    r"catamenial", r"time of the month", r"before (?:her )?(?:periods?|menses)", r"\bmenstrual\b", r"each month", r"every month", r"when bleeding", r"monthly",
]
MENSTRUAL_LINK_NO = [
    r"(?:not |un)related to (?:her )?(?:periods?|menses|cycle)", r"no (?:relation|relationship|link|association) (?:to|with) (?:her )?(?:periods?|menses|cycle)",
    r"no cyclical (?:pattern|component)", r"non-?cyclical", r"independent of (?:her )?(?:periods?|cycle)", r"throughout the (?:month|cycle)",
    r"not (?:linked|associated) (?:to|with) (?:her )?(?:periods?|menses|cycle)",
]
SEVERE_CUES = [r"\bsevere\b", r"\bexcruciating\b", r"\b(?:8|9|10)/10\b", r"\bdebilitating\b", r"\bcrippling\b", r"\bterrible\b", r"\bintense\b"]
MODERATE_CUES = [r"\bmoderate\b", r"\b(?:5|6|7)/10\b"]
MILD_CUES = [r"\bmild\b", r"\b(?:1|2|3|4)/10\b", r"\bslight\b"]
IMPACT_CUES = [r"(?:missed|missing|off) (?:work|school|uni|college|classes)", r"time off (?:work|school)", r"bed-?bound", r"in bed for", r"unable to (?:work|function|go to school)", r"affect(?:s|ing) (?:her )?(?:work|daily|studies|life)", r"called in sick", r"\bA&E\b"]
FAMILY_HISTORY_PATTERNS = [r"(?:mother|mum|sister|aunt)\b[^.]{0,40}endometriosis", r"endometriosis[^.]{0,30}(?:mother|mum|sister|aunt)", r"FH(?: of)? endometriosis", r"family history of endometriosis"]

# Pseudo-negations: contain a negation word but do NOT negate the symptom (NegEx-style).
PSEUDO_NEGATION = [r"no improvement", r"no better", r"no change", r"no relief", r"not improved", r"not settled",
                   r"not resolved", r"not helping", r"not helped", r"not responding", r"no response", r"not controlled",
                   r"not much better", r"no significant improvement", r"not sure"]
TREATMENT_FAILURE = [r"no improvement", r"no better", r"no relief", r"not (?:helping|helped|improved|settled|controlled|responding)",
                     r"no response", r"(?:ibuprofen|naproxen|mefenamic|pill|COCP|analgesia|painkillers)[^.]{0,30}(?:ineffective|not working|doesn'?t help|did(?:n't| not) help)",
                     r"persist(?:s|ing|ent) despite", r"despite (?:the )?(?:pill|COCP|NSAIDs?|ibuprofen|mefenamic|naproxen|treatment)", r"ongoing despite"]
TREATMENT_SUCCESS = [r"much (?:better|improved)", r"well controlled", r"(?:pain|symptoms?) (?:has |have )?(?:settled|resolved)", r"good response", r"significant(?:ly)? improve", r"happy with (?:the )?(?:pill|treatment)"]
