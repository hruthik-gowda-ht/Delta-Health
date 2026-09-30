"""
stock_advisor.py - "we detected a spike, here's what to stock up on."

Two signals feed this:
1. Per-clinic outbreak spikes (from alerts.check_local_spike) combined with
   whatever condition_tags doctors have logged recently at that clinic, to
   infer a likely disease pattern and recommend disease-specific supplies.
2. Network-wide treatment usage trends (from TreatmentRecord logging
   frequency) as a proxy for consumption -- a treatment being logged much
   more often this week than last week is a rising-demand signal, useful
   for a kirana-style pharmacy deciding what to reorder.

Like differential.py, this is transparent rule-based decision support, not
a trained demand-forecasting model -- we don't have real sales/stock data
to train one honestly. Recommendations are directional ("consider more"),
not prescriptive quantities.
"""
import datetime
from sqlalchemy.orm import Session

from . import models

# disease name -> supplies a clinic/pharmacy would want more of, matching
# the disease names used in differential.py so the two stay consistent
SUPPLY_MAP = {
    "Dengue (suspected)": [
        "NS1 antigen / rapid test kits", "Paracetamol (fever, non-NSAID)",
        "Oral rehydration salts (ORS) / IV fluids", "Platelet count test strips",
    ],
    "Chikungunya (suspected)": [
        "Paracetamol (fever, non-NSAID)", "Topical anti-inflammatory (non-aspirin)",
        "Oral rehydration salts (ORS)",
    ],
    "Typhoid (suspected)": [
        "Blood culture / Widal test kits", "Antibiotics per local protocol",
        "Oral rehydration salts (ORS) / IV fluids",
    ],
    "Malaria (suspected)": [
        "Rapid diagnostic test (RDT) kits", "Antimalarials per local protocol",
        "IV fluids",
    ],
    "Viral fever / URI (suspected)": [
        "Paracetamol", "Cough syrup / lozenges", "Oral rehydration salts (ORS)",
    ],
}

GENERIC_SUPPLIES = [
    "Paracetamol", "Oral rehydration salts (ORS)", "Basic fever diagnostic kits",
    "Thermometers / consumables",
]

# keywords a doctor might type in condition_tag -> which disease bucket it maps to
CONDITION_KEYWORDS = {
    "Dengue (suspected)": ["dengue"],
    "Chikungunya (suspected)": ["chikungunya", "chikun"],
    "Typhoid (suspected)": ["typhoid"],
    "Malaria (suspected)": ["malaria"],
    "Viral fever / URI (suspected)": ["viral", "flu", "resp", "cold", "uri", "cough"],
}

STOCK_LOOKBACK_DAYS = 3   # how recent a condition_tag mention must be to count
TREND_WINDOW_DAYS = 7     # compare last 7 days vs previous 7 days of usage


def _infer_disease_for_clinic(db: Session, clinic_id: int):
    """Looks at this clinic's recently logged treatment condition_tags and
    tries to match them to a known disease bucket. Returns (disease_name,
    matched_tag, mention_count) or (None, None, 0) if nothing matches."""
    since = datetime.datetime.utcnow() - datetime.timedelta(days=STOCK_LOOKBACK_DAYS)
    records = (db.query(models.TreatmentRecord)
               .filter(models.TreatmentRecord.clinic_id == clinic_id,
                       models.TreatmentRecord.created_at >= since)
               .all())
    if not records:
        return None, None, 0

    tallies = {}
    for disease, keywords in CONDITION_KEYWORDS.items():
        count = sum(1 for r in records if any(k in r.condition_tag.lower() for k in keywords))
        if count:
            tallies[disease] = count

    if not tallies:
        return None, None, 0

    best_disease = max(tallies, key=tallies.get)
    return best_disease, best_disease, tallies[best_disease]


def get_clinic_advisories(db: Session):
    """One advisory per clinic that currently has an active symptom spike
    (last 24h), recommending specific supplies based on whatever disease
    pattern is inferable from that clinic's recent treatment logs, or
    generic fever-cluster supplies if nothing more specific is known."""
    since_24h = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    spike_alerts = (db.query(models.Alert)
                     .filter(models.Alert.alert_type == "local_spike",
                             models.Alert.created_at >= since_24h)
                     .all())
    spiking_clinic_ids = list({a.clinic_id for a in spike_alerts if a.clinic_id})
    if not spiking_clinic_ids:
        return []

    clinics = {c.id: c for c in db.query(models.Clinic)
               .filter(models.Clinic.id.in_(spiking_clinic_ids)).all()}

    advisories = []
    for clinic_id, clinic in clinics.items():
        disease, matched_tag, mentions = _infer_disease_for_clinic(db, clinic_id)
        if disease:
            supplies = SUPPLY_MAP[disease]
            reason = (f"{mentions} recent treatment log(s) at this clinic mention '{matched_tag}' "
                      f"-- pattern matches {disease}.")
        else:
            supplies = GENERIC_SUPPLIES
            reason = ("Fever-symptom case volume is elevated but no specific disease pattern has "
                      "been logged yet -- general fever-cluster supplies recommended.")

        advisories.append({
            "clinic_id": clinic_id, "clinic_name": clinic.name, "city": clinic.city,
            "likely_pattern": disease or "Unspecified fever cluster",
            "reason": reason, "recommended_supplies": supplies,
        })

    return advisories


def get_network_demand_trends(db: Session):
    """Network-wide: which treatments are being logged much more often this
    week than last week -- a proxy for rising consumption/demand, useful
    for restocking decisions even at clinics that aren't currently spiking."""
    now = datetime.datetime.utcnow()
    this_window_start = now - datetime.timedelta(days=TREND_WINDOW_DAYS)
    prev_window_start = now - datetime.timedelta(days=TREND_WINDOW_DAYS * 2)

    recent = (db.query(models.TreatmentRecord)
              .filter(models.TreatmentRecord.created_at >= this_window_start).all())
    previous = (db.query(models.TreatmentRecord)
                .filter(models.TreatmentRecord.created_at >= prev_window_start,
                        models.TreatmentRecord.created_at < this_window_start).all())

    def tally(records):
        counts = {}
        for r in records:
            key = r.treatment_name.strip().lower()
            counts[key] = counts.get(key, 0) + 1
        return counts

    recent_counts = tally(recent)
    prev_counts = tally(previous)

    trends = []
    for name, count in recent_counts.items():
        prev = prev_counts.get(name, 0)
        if count < 3:
            continue  # not enough signal
        if prev == 0:
            change_pct = 100.0 if count >= 3 else 0.0
        else:
            change_pct = round(((count - prev) / prev) * 100, 1)
        trends.append({
            "treatment_name": name, "this_week_count": count,
            "previous_week_count": prev, "change_pct": change_pct,
        })

    trends.sort(key=lambda t: -t["change_pct"])
    return trends
