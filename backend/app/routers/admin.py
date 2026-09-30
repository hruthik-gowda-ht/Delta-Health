import json
import datetime
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from .. import models, schemas, fl_core, alerts, stock_advisor
from ..deps import get_db, require_admin_user

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/overview", response_model=schemas.AdminOverview)
def overview(db: Session = Depends(get_db), user: models.User = Depends(require_admin_user)):
    state = fl_core.get_or_create_model_state(db)
    total_clinics = db.query(func.count(models.Clinic.id)).scalar() or 0
    total_patients = db.query(func.count(models.Patient.id)).scalar() or 0
    pending = (db.query(func.count(models.DeltaSubmission.id))
               .filter(models.DeltaSubmission.round == state.round + 1,
                       models.DeltaSubmission.applied == False).scalar() or 0)  # noqa: E712

    history = (db.query(models.TrainingRound)
               .order_by(models.TrainingRound.round_number.asc()).all())
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


@router.post("/aggregate")
def aggregate(db: Session = Depends(get_db), user: models.User = Depends(require_admin_user)):
    state = fl_core.get_or_create_model_state(db)
    target_round = state.round + 1

    submissions = (db.query(models.DeltaSubmission)
                   .filter(models.DeltaSubmission.round == target_round,
                           models.DeltaSubmission.applied == False)  # noqa: E712
                   .all())
    if not submissions:
        raise HTTPException(status_code=400, detail="No pending delta submissions to aggregate yet")

    deltas = [np.array(json.loads(s.delta_json)) for s in submissions]
    counts = [s.n_samples for s in submissions]

    deltas = fl_core.clip_outlier_deltas(deltas)
    agg_delta = fl_core.federated_average(deltas, counts)

    weights = np.array(json.loads(state.weights_json)) + agg_delta
    mean = json.loads(state.mean_json)
    std = json.loads(state.std_json)

    acc = fl_core.evaluate_weights(weights.tolist(), mean, std, fl_core.HOLDOUT_X, fl_core.HOLDOUT_Y)

    state.weights_json = json.dumps(weights.tolist())
    state.round = target_round
    state.updated_at = datetime.datetime.utcnow()

    db.add(models.TrainingRound(round_number=target_round, global_accuracy=acc,
                                 num_clinics=len(submissions)))
    for s in submissions:
        s.applied = True
    db.commit()

    return {
        "round": target_round,
        "global_accuracy": round(acc, 4),
        "clinics_included": len(submissions),
        "clinic_ids": [s.clinic_id for s in submissions],
    }


@router.get("/heatmap", response_model=List[schemas.ClinicHeatmapPoint])
def heatmap(db: Session = Depends(get_db), user: models.User = Depends(require_admin_user)):
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
def all_alerts(limit: int = 30, db: Session = Depends(get_db), user: models.User = Depends(require_admin_user)):
    return (db.query(models.Alert).order_by(models.Alert.created_at.desc()).limit(limit).all())


@router.post("/alerts/check_regional", response_model=schemas.RegionalCheckResult)
def trigger_regional_check(db: Session = Depends(get_db), user: models.User = Depends(require_admin_user)):
    alert, diag = alerts.check_regional_outbreak(db)
    return schemas.RegionalCheckResult(
        alert_created=alert.message if alert else None,
        spiking_clinics=[schemas.SpikingClinic(**c) for c in diag["spiking_clinics"]],
        largest_cluster_size=diag["largest_cluster_size"],
        min_clinics_required=diag["min_clinics_required"],
        radius_km=diag["radius_km"],
    )


@router.get("/clinics")
def list_clinics(db: Session = Depends(get_db), user: models.User = Depends(require_admin_user)):
    clinics = db.query(models.Clinic).all()
    out = []
    for c in clinics:
        total = db.query(func.count(models.Patient.id)).filter(models.Patient.clinic_id == c.id).scalar() or 0
        out.append({"id": c.id, "name": c.name, "city": c.city, "lat": c.lat, "lng": c.lng,
                    "patient_count": total})
    return out


@router.get("/treatments", response_model=List[schemas.TreatmentEfficacyRow])
def treatment_efficacy(db: Session = Depends(get_db), user: models.User = Depends(require_admin_user)):
    """Feature #3 (public health view): drug efficacy across the whole
    network, grouped by treatment + condition. This is the 'antimicrobial
    resistance noticed in days, not months' dashboard."""
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


@router.post("/alerts/check_treatment_failures", response_model=schemas.TreatmentCheckResult)
def trigger_treatment_failure_check(db: Session = Depends(get_db), user: models.User = Depends(require_admin_user)):
    created, diagnostics = alerts.check_treatment_failures(db)
    return schemas.TreatmentCheckResult(
        alerts_created=[a.message for a in created],
        diagnostics=[schemas.TreatmentDiagnostic(**d) for d in diagnostics],
        min_records=alerts.TREATMENT_FAILURE_MIN_RECORDS,
        min_clinics=alerts.TREATMENT_FAILURE_MIN_CLINICS,
        rate_threshold=alerts.TREATMENT_FAILURE_RATE_THRESHOLD,
    )


@router.get("/outbreak_timeline", response_model=schemas.OutbreakTimelineResponse)
def outbreak_timeline(days: int = 14, db: Session = Depends(get_db),
                       user: models.User = Depends(require_admin_user)):
    """Powers the Outbreak Monitor page: daily fever-symptom case counts
    per clinic, for charting, plus which clinics are currently spiking so
    the map/chart can highlight them."""
    days = max(3, min(60, days))
    today = datetime.datetime.utcnow().date()
    date_list = [(today - datetime.timedelta(days=i)) for i in range(days - 1, -1, -1)]
    date_strs = [d.isoformat() for d in date_list]

    since = datetime.datetime.combine(date_list[0], datetime.time.min)
    patients = (db.query(models.Patient)
                .filter(models.Patient.created_at >= since, models.Patient.fever_days >= 1)
                .all())

    counts_by_clinic: dict = {}
    for p in patients:
        d = p.created_at.date()
        counts_by_clinic.setdefault(p.clinic_id, {}).setdefault(d, 0)
        counts_by_clinic[p.clinic_id][d] += 1

    since_24h = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    spiking_ids = {a.clinic_id for a in db.query(models.Alert)
                   .filter(models.Alert.alert_type == "local_spike", models.Alert.created_at >= since_24h)
                   .all() if a.clinic_id}

    clinics = db.query(models.Clinic).all()
    series = []
    for c in clinics:
        daily = [counts_by_clinic.get(c.id, {}).get(d, 0) for d in date_list]
        series.append(schemas.ClinicTimelineSeries(
            clinic_id=c.id, name=c.name, city=c.city, lat=c.lat, lng=c.lng,
            is_spiking=c.id in spiking_ids, daily_counts=daily,
        ))

    return schemas.OutbreakTimelineResponse(
        dates=date_strs, series=series,
        radius_km=alerts.REGIONAL_ALERT_RADIUS_KM, min_clinics_required=alerts.REGIONAL_ALERT_MIN_CLINICS,
    )


@router.get("/drug_efficacy_timeline", response_model=schemas.DrugEfficacyTimelineResponse)
def drug_efficacy_timeline(days: int = 30, db: Session = Depends(get_db),
                            user: models.User = Depends(require_admin_user)):
    """Powers the Drug Efficacy Tracker page: daily outcome counts per
    treatment+condition combo, for trend charting."""
    days = max(3, min(90, days))
    today = datetime.datetime.utcnow().date()
    date_list = [(today - datetime.timedelta(days=i)) for i in range(days - 1, -1, -1)]
    date_strs = [d.isoformat() for d in date_list]
    date_index = {d: i for i, d in enumerate(date_list)}

    since = datetime.datetime.combine(date_list[0], datetime.time.min)
    records = (db.query(models.TreatmentRecord)
               .filter(models.TreatmentRecord.created_at >= since)
               .all())

    groups: dict = {}
    for r in records:
        key = (r.treatment_name.strip(), r.condition_tag.strip())
        groups.setdefault(key, []).append(r)

    series = []
    for (treatment, condition), recs in groups.items():
        improved = [0] * days
        no_change = [0] * days
        worsened = [0] * days
        for r in recs:
            idx = date_index.get(r.created_at.date())
            if idx is None:
                continue
            if r.outcome == "improved":
                improved[idx] += 1
            elif r.outcome == "no_change":
                no_change[idx] += 1
            elif r.outcome == "worsened":
                worsened[idx] += 1
        failures = sum(1 for r in recs if r.outcome in ("no_change", "worsened"))
        series.append(schemas.TreatmentTimelineSeries(
            treatment=treatment, condition=condition, total_records=len(recs),
            failure_rate=round(failures / len(recs), 3) if recs else 0.0,
            daily_improved=improved, daily_no_change=no_change, daily_worsened=worsened,
        ))
    series.sort(key=lambda s: -s.failure_rate)

    return schemas.DrugEfficacyTimelineResponse(dates=date_strs, series=series)


@router.get("/stock_advisory", response_model=schemas.StockAdvisoryResponse)
def stock_advisory(db: Session = Depends(get_db), user: models.User = Depends(require_admin_user)):
    """Pharmacy/clinic stock advisory: per-clinic disease-specific supply
    recommendations for currently-spiking clinics, plus network-wide
    treatment demand trends (rising usage = reorder signal)."""
    clinic_advisories = stock_advisor.get_clinic_advisories(db)
    demand_trends = stock_advisor.get_network_demand_trends(db)
    return schemas.StockAdvisoryResponse(
        clinic_advisories=[schemas.ClinicStockAdvisory(**a) for a in clinic_advisories],
        demand_trends=[schemas.DemandTrend(**t) for t in demand_trends],
    )
