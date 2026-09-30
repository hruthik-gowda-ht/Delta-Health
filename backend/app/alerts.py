"""
alerts.py - the "hyper-local early warning system" feature.

Two detectors:
1. check_local_spike: does THIS clinic's symptom volume today look like an
   anomaly vs its own recent baseline?
2. check_regional_outbreak: are MULTIPLE nearby clinics spiking on the same
   symptom pattern at the same time? That's the "45 clinics in a 10km
   radius" scenario from the pitch -- a signal no single clinic could see
   on its own, which only exists because of the federated network.
"""
import math
import datetime
from typing import Optional
from sqlalchemy.orm import Session

from . import models
from .config import (OUTBREAK_SPIKE_RATIO, OUTBREAK_MIN_COUNT,
                      REGIONAL_ALERT_RADIUS_KM, REGIONAL_ALERT_MIN_CLINICS,
                      TREATMENT_FAILURE_MIN_RECORDS, TREATMENT_FAILURE_MIN_CLINICS,
                      TREATMENT_FAILURE_RATE_THRESHOLD, TREATMENT_FAILURE_WINDOW_DAYS)


def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _symptomatic_count(db: Session, clinic_id: int, since: datetime.datetime,
                        until: Optional[datetime.datetime] = None):
    q = db.query(models.Patient).filter(
        models.Patient.clinic_id == clinic_id,
        models.Patient.created_at >= since,
        models.Patient.fever_days >= 1,
    )
    if until:
        q = q.filter(models.Patient.created_at < until)
    return q.count()


def check_local_spike(db: Session, clinic_id: int) -> Optional[models.Alert]:
    """Compares today's fever-symptom patient count against the clinic's
    own 7-day trailing average. Fires an Alert row if it looks like a spike."""
    now = datetime.datetime.utcnow()
    today_start = now - datetime.timedelta(hours=24)
    baseline_start = now - datetime.timedelta(days=8)

    today_count = _symptomatic_count(db, clinic_id, today_start)
    baseline_total = _symptomatic_count(db, clinic_id, baseline_start, today_start)
    baseline_avg = baseline_total / 7.0

    if today_count < OUTBREAK_MIN_COUNT:
        return None
    if baseline_avg > 0 and today_count / baseline_avg < OUTBREAK_SPIKE_RATIO:
        return None
    if baseline_avg == 0 and today_count < OUTBREAK_MIN_COUNT + 2:
        return None

    ratio_txt = f"{today_count / baseline_avg:.1f}x" if baseline_avg > 0 else f"{today_count} new"
    clinic = db.get(models.Clinic, clinic_id)
    alert = models.Alert(
        clinic_id=clinic_id,
        alert_type="local_spike",
        severity="warning",
        message=(f"{clinic.name}: fever-symptom cases up {ratio_txt} vs the trailing "
                 f"7-day average ({today_count} cases in the last 24h)."),
    )
    db.add(alert)
    db.commit()
    return alert


def check_regional_outbreak(db: Session) -> Optional[models.Alert]:
    """Looks at ALL clinics with a local_spike alert in the last 24h and
    checks if enough of them sit within REGIONAL_ALERT_RADIUS_KM of each
    other -- this is only detectable because the network connects clinics.

    Returns (alert_or_None, diagnostics) where diagnostics always describes
    the current state (which clinics are spiking, largest nearby cluster
    found) so a caller can show progress even when nothing fires."""
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    recent_spikes = (db.query(models.Alert)
                      .filter(models.Alert.alert_type == "local_spike",
                              models.Alert.created_at >= since)
                      .all())
    spiking_clinic_ids = list({a.clinic_id for a in recent_spikes if a.clinic_id})

    diagnostics = {
        "spiking_clinics": [], "largest_cluster_size": 0,
        "min_clinics_required": REGIONAL_ALERT_MIN_CLINICS, "radius_km": REGIONAL_ALERT_RADIUS_KM,
    }

    if not spiking_clinic_ids:
        return None, diagnostics

    clinics = {c.id: c for c in db.query(models.Clinic).filter(models.Clinic.id.in_(spiking_clinic_ids)).all()}
    diagnostics["spiking_clinics"] = [{"name": c.name, "city": c.city} for c in clinics.values()]

    if len(spiking_clinic_ids) < REGIONAL_ALERT_MIN_CLINICS:
        return None, diagnostics

    best_cluster = []
    for cid, clinic in clinics.items():
        cluster = [c2 for c2id, c2 in clinics.items()
                   if haversine_km(clinic.lat, clinic.lng, c2.lat, c2.lng) <= REGIONAL_ALERT_RADIUS_KM]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    diagnostics["largest_cluster_size"] = len(best_cluster)

    if len(best_cluster) < REGIONAL_ALERT_MIN_CLINICS:
        return None, diagnostics

    names = ", ".join(c.name for c in best_cluster)
    alert = models.Alert(
        clinic_id=None,
        alert_type="regional_outbreak",
        severity="critical",
        message=(f"Regional signal: {len(best_cluster)} clinics within "
                 f"{REGIONAL_ALERT_RADIUS_KM:.0f}km of each other are seeing simultaneous "
                 f"fever-symptom spikes ({names}). Possible localized outbreak."),
    )
    db.add(alert)
    db.commit()
    return alert, diagnostics



def check_treatment_failures(db: Session):
    """Drug efficacy / antimicrobial resistance signal: groups recent
    TreatmentRecord rows by (treatment_name, condition_tag) ACROSS ALL
    CLINICS. If enough recent outcomes for the same treatment+condition
    combo are 'no_change' or 'worsened', across multiple clinics, that's
    the 'this antibiotic stopped working and nobody noticed for months'
    problem from the pitch -- surfaced in days instead, because the
    network can see patterns no single clinic could see alone.

    Returns (created_alerts, diagnostics) where diagnostics describes the
    closest-to-triggering group even when nothing fired, so a caller can
    explain WHY instead of just saying 'nothing detected'.
    """
    since = datetime.datetime.utcnow() - datetime.timedelta(days=TREATMENT_FAILURE_WINDOW_DAYS)
    records = (db.query(models.TreatmentRecord)
               .filter(models.TreatmentRecord.created_at >= since)
               .all())

    groups: dict = {}
    for r in records:
        key = (r.treatment_name.strip().lower(), r.condition_tag.strip().lower())
        groups.setdefault(key, []).append(r)

    created = []
    diagnostics = []
    for (treatment, condition), rows in groups.items():
        clinics_involved = {r.clinic_id for r in rows}
        failures = sum(1 for r in rows if r.outcome in ("no_change", "worsened"))
        failure_rate = failures / len(rows) if rows else 0.0
        diagnostics.append({
            "treatment": treatment, "condition": condition, "records": len(rows),
            "clinics": len(clinics_involved), "failure_rate": round(failure_rate, 3),
        })

        if len(rows) < TREATMENT_FAILURE_MIN_RECORDS:
            continue
        if len(clinics_involved) < TREATMENT_FAILURE_MIN_CLINICS:
            continue
        if failure_rate < TREATMENT_FAILURE_RATE_THRESHOLD:
            continue

        # avoid spamming a duplicate alert every time this check reruns soon after
        recent_dup = (db.query(models.Alert)
                      .filter(models.Alert.alert_type == "treatment_failure",
                              models.Alert.message.contains(treatment),
                              models.Alert.created_at >= datetime.datetime.utcnow() - datetime.timedelta(hours=12))
                      .first())
        if recent_dup:
            continue

        alert = models.Alert(
            clinic_id=None,
            alert_type="treatment_failure",
            severity="critical",
            message=(f"Possible treatment failure signal: '{treatment}' for '{condition}' shows a "
                     f"{failure_rate*100:.0f}% non-improvement rate across {len(rows)} recent cases "
                     f"from {len(clinics_involved)} clinics (last {TREATMENT_FAILURE_WINDOW_DAYS} days). "
                     f"Consider reviewing for resistance."),
        )
        db.add(alert)
        created.append(alert)

    if created:
        db.commit()

    # closest candidate = most clinics represented, tie-broken by most records
    diagnostics.sort(key=lambda d: (-d["clinics"], -d["records"]))
    return created, diagnostics[:3]
