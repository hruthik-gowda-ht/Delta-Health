"""
differential.py - a transparent, rule-based differential diagnosis layer.

IMPORTANT DESIGN NOTE: this is deliberately NOT a trained ML classifier.
We don't have per-disease labeled patient data to train one honestly, and
faking that with fabricated training would be worse than being upfront
about it. This module scores textbook symptom clusters (WHO-style clinical
patterns) against what the doctor observed, in symptoms the doctor
directly reports vs the actual global logistic-regression risk model
(fl_core.py), which IS real federated ML. Keep these two clearly separate
in the API and UI: "risk score" = real model, "differential diagnosis" =
rule-based decision support.

None of the recommendations below include drug dosages -- only which
tests/classes of action a doctor would consider, consistent with standard
public clinical guidelines (e.g. WHO dengue guidance on avoiding NSAIDs).
This is decision support for a licensed clinician already treating the
patient, not standalone diagnostic or self-treatment advice.
"""

SYMPTOM_LABELS = {
    "high_fever": "High fever (>102°F)",
    "pain_behind_eyes": "Pain behind the eyes",
    "dry_cough": "Dry cough & cold",
    "low_platelet": "Low platelet count (<100k)",
    "joint_bone_pain": "Severe joint / bone pain",
    "skin_rash": "Skin rash / petechiae",
    "nausea_vomiting": "Nausea / vomiting",
    "diarrhea_abdominal": "Diarrhea / abdominal cramps",
    "chills_rigors": "Chills & rigors",
    "sore_throat": "Sore throat",
}

DISEASES = {
    "Dengue (suspected)": {
        "weights": {"high_fever": 3, "pain_behind_eyes": 3, "low_platelet": 4,
                     "skin_rash": 1.5, "nausea_vomiting": 1},
        "fever_bonus": lambda days: 0.5 if days <= 6 else -1.0,
        "recommendations": [
            "Order NS1 antigen and IgM/IgG rapid tests to confirm.",
            "Favor oral or IV rehydration per standard fluid protocol.",
            "Avoid NSAIDs/aspirin given bleeding risk; paracetamol is generally preferred for fever.",
            "Watch for warning signs: persistent vomiting, abdominal pain, mucosal bleeding, lethargy — escalate if present.",
        ],
    },
    "Chikungunya (suspected)": {
        "weights": {"joint_bone_pain": 4, "high_fever": 2, "skin_rash": 1.5, "pain_behind_eyes": 0.5},
        "fever_bonus": lambda days: 0.0,
        "recommendations": [
            "Symptomatic management with rest and hydration.",
            "Avoid aspirin/NSAIDs until dengue is ruled out, given overlapping presentation.",
            "Expect joint pain may persist beyond fever resolution; note for follow-up.",
        ],
    },
    "Typhoid (suspected)": {
        "weights": {"diarrhea_abdominal": 2.5, "nausea_vomiting": 1, "high_fever": 1.5},
        "fever_bonus": lambda days: 2.0 if days >= 7 else 0.0,
        "recommendations": [
            "Consider blood culture (and Widal test where culture isn't available) to confirm.",
            "Initiate empiric antibiotic therapy per local antimicrobial guidelines if clinically indicated.",
            "Monitor hydration status and for signs of intestinal complications.",
        ],
    },
    "Malaria (suspected)": {
        "weights": {"chills_rigors": 4, "high_fever": 2, "nausea_vomiting": 1},
        "fever_bonus": lambda days: 1.0 if days >= 7 else 0.0,
        "recommendations": [
            "Order a rapid diagnostic test or blood smear for confirmation.",
            "Initiate antimalarial therapy per local protocol if confirmed or strongly suspected.",
            "Monitor for severe malaria signs: altered consciousness, repeated seizures, respiratory distress.",
        ],
    },
    "Viral fever / URI (suspected)": {
        "weights": {"dry_cough": 3, "sore_throat": 3, "high_fever": 1},
        "fever_bonus": lambda days: 1.0 if days <= 3 else -1.0,
        "recommendations": [
            "Supportive care: rest, fluids, symptomatic fever management.",
            "Reassess if symptoms persist beyond 5-7 days or worsen — consider bacterial superinfection.",
        ],
    },
}


def evaluate_differential(symptoms: dict, fever_days: float):
    """symptoms: dict of the SYMPTOM_LABELS keys -> bool.
    Returns a list of {disease, match_pct, reasons, recommendations},
    sorted by match_pct descending. match_pct is independent per disease
    (how well this presentation matches THAT disease's textbook pattern),
    not a shared probability distribution across diseases -- so they do
    not need to sum to 100, matching how differential diagnosis actually
    works (a patient can plausibly match more than one pattern at once).
    """
    results = []
    for name, cfg in DISEASES.items():
        weights = cfg["weights"]
        score = sum(w for k, w in weights.items() if symptoms.get(k))
        score += cfg["fever_bonus"](fever_days)
        max_score = sum(w for w in weights.values() if w > 0)
        match_pct = max(0.0, min(100.0, (score / max_score) * 100)) if max_score > 0 else 0.0
        reasons = [SYMPTOM_LABELS[k] for k, w in weights.items() if symptoms.get(k) and w > 0]
        results.append({
            "disease": name, "match_pct": round(match_pct, 1),
            "reasons": reasons, "recommendations": cfg["recommendations"],
        })
    results.sort(key=lambda r: -r["match_pct"])
    return results
