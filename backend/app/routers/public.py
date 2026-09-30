"""
public.py - the PUBLIC health dashboard API. No authentication required.

CRITICAL INVARIANT: every endpoint here must return only aggregate,
de-identified data -- the same privacy bar as the admin endpoints, just
reachable without a login so press, other health officials, or the public
can see district-wide signals. Never expose a Patient row, a symptom_tags
string tied to an individual, or anything that could re-identify a person.
Clinic-level aggregates (case counts, accuracy, alerts) are fine to expose
publicly, the same way real public health dashboards show hospital-level
admission counts without patient names.
"""
import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from .. import models, schemas, fl_core, stock_advisor
from ..deps import get_db

router = APIRouter(prefix="/public", tags=["public"])


@router.get("/overview", response_model=schemas.AdminOverview)
def public_overview(db: Session = Depends(get_db)):
    state = fl_core.get_or_create_model_state(db)
    total_clinics = db.query(func.count(models.Clinic.id)).scalar() or 0
    total_patients = db.query(func.count(models.Patient.id)).scalar() or 0
    pending = (db.query(func.count(models.DeltaSubmission.id))
               .filter(models.DeltaSubmission.round == state.round + 1,
                       models.DeltaSubmission.applied == False).scalar() or 0)  # noqa: E712

    history = (db.query(models.TrainingRound).order_by(models.TrainingRound.round_number.asc()).all())
    last_round = history[-1] if history else None

    return schemas.AdminOverview(
        current_round=state.round,
        global_accuracy=last_round.global_accuracy if last_round else None,
        total_clinics=total_clinics,
        total_patients=total_patients,
        pending_submissions=pending,
        accuracy_history=[{"round": h.round_number, "accuracy": h.global_accuracy,
                            "num_clinics": h.num_clinics} for h in history],
    )


@router.get("/heatmap", response_model=List[schemas.ClinicHeatmapPoint])
def public_heatmap(db: Session = Depends(get_db)):
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    clinics = db.query(models.Clinic).all()
    points = []
    for c in clinics:
        total = db.query(func.count(models.Patient.id)).filter(models.Patient.clinic_id == c.id).scalar() or 0
        at_risk = (db.query(func.count(models.Patient.id))
                   .filter(models.Patient.clinic_id == c.id, models.Patient.at_risk == 1).scalar() or 0)
        alert_count = (db.query(func.count(models.Alert.id))
                       .filter(models.Alert.clinic_id == c.id, models.Alert.created_at >= since).scalar() or 0)
        points.append(schemas.ClinicHeatmapPoint(
            clinic_id=c.id, name=c.name, city=c.city, lat=c.lat, lng=c.lng,
            patient_count=total, at_risk_rate=round((at_risk / total) if total else 0.0, 3),
            alert_count_24h=alert_count,
        ))
    return points


@router.get("/alerts", response_model=List[schemas.AlertOut])
def public_alerts(limit: int = 20, db: Session = Depends(get_db)):
    return (db.query(models.Alert).order_by(models.Alert.created_at.desc()).limit(limit).all())


@router.get("/treatments", response_model=List[schemas.TreatmentEfficacyRow])
def public_treatment_efficacy(db: Session = Depends(get_db)):
    records = db.query(models.TreatmentRecord).all()
    groups: dict = {}
    for r in records:
        key = (r.treatment_name.strip(), r.condition_tag.strip())
        groups.setdefault(key, []).append(r)

    rows = []
    for (treatment, condition), recs in groups.items():
        failures = sum(1 for r in recs if r.outcome in ("no_change", "worsened"))
        rows.append(schemas.TreatmentEfficacyRow(
            treatment_name=treatment, condition_tag=condition,
            total_records=len(recs), clinics_represented=len({r.clinic_id for r in recs}),
            failure_rate=round(failures / len(recs), 3),
        ))
    rows.sort(key=lambda r: -r.failure_rate)
    return rows


@router.get("/clinics")
def public_clinics(db: Session = Depends(get_db)):
    clinics = db.query(models.Clinic).all()
    out = []
    for c in clinics:
        total = db.query(func.count(models.Patient.id)).filter(models.Patient.clinic_id == c.id).scalar() or 0
        out.append({"id": c.id, "name": c.name, "city": c.city, "lat": c.lat, "lng": c.lng,
                    "patient_count": total})
    return out


@router.get("/stock_advisory", response_model=schemas.StockAdvisoryResponse)
def public_stock_advisory(db: Session = Depends(get_db)):
    """Public version -- same aggregate-only data as the admin endpoint.
    Deliberately not clinic-identity-sensitive: which supplies a clinic
    might need is not private in the way patient records are, and kirana
    pharmacies without a login should be able to see network demand trends."""
    clinic_advisories = stock_advisor.get_clinic_advisories(db)
    demand_trends = stock_advisor.get_network_demand_trends(db)
    return schemas.StockAdvisoryResponse(
        clinic_advisories=[schemas.ClinicStockAdvisory(**a) for a in clinic_advisories],
        demand_trends=[schemas.DemandTrend(**t) for t in demand_trends],
    )
