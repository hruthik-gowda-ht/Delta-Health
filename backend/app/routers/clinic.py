import json
import csv
import io
import datetime
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from .. import models, schemas, fl_core, alerts, differential
from ..deps import get_db, require_clinic_user

router = APIRouter(prefix="/clinic", tags=["clinic"])


def _clinic_id(user: models.User) -> int:
    if not user.clinic_id:
        raise HTTPException(status_code=400, detail="This account is not linked to a clinic")
    return user.clinic_id


@router.post("/patients", response_model=schemas.PatientOut)
def add_patient(payload: schemas.PatientCreate, db: Session = Depends(get_db),
                 user: models.User = Depends(require_clinic_user)):
    clinic_id = _clinic_id(user)
    patient = models.Patient(clinic_id=clinic_id, **payload.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)

    # fire the hyper-local early warning check right after each new entry
    alerts.check_local_spike(db, clinic_id)
    return patient


@router.get("/patients", response_model=List[schemas.PatientOut])
def list_patients(limit: int = 50, db: Session = Depends(get_db),
                   user: models.User = Depends(require_clinic_user)):
    clinic_id = _clinic_id(user)
    return (db.query(models.Patient)
            .filter(models.Patient.clinic_id == clinic_id)
            .order_by(models.Patient.created_at.desc())
            .limit(limit).all())


@router.get("/summary", response_model=schemas.ClinicSummary)
def summary(db: Session = Depends(get_db), user: models.User = Depends(require_clinic_user)):
    clinic_id = _clinic_id(user)
    clinic = db.get(models.Clinic, clinic_id)
    state = fl_core.get_or_create_model_state(db)

    total = db.query(func.count(models.Patient.id)).filter(models.Patient.clinic_id == clinic_id).scalar() or 0
    at_risk = (db.query(func.count(models.Patient.id))
               .filter(models.Patient.clinic_id == clinic_id, models.Patient.at_risk == 1).scalar() or 0)
    at_risk_rate = (at_risk / total) if total else 0.0

    last_sub = (db.query(models.DeltaSubmission)
                .filter(models.DeltaSubmission.clinic_id == clinic_id)
                .order_by(models.DeltaSubmission.created_at.desc()).first())

    submitted_this_round = (db.query(models.DeltaSubmission)
                             .filter(models.DeltaSubmission.clinic_id == clinic_id,
                                     models.DeltaSubmission.round == state.round + 1,
                                     models.DeltaSubmission.applied == False)  # noqa: E712
                             .first() is not None)

    last_round = (db.query(models.TrainingRound)
                  .order_by(models.TrainingRound.round_number.desc()).first())

    return schemas.ClinicSummary(
        clinic_id=clinic_id, clinic_name=clinic.name, patient_count=total,
        at_risk_rate=round(at_risk_rate, 3),
        last_local_acc=last_sub.local_acc if last_sub else None,
        global_round=state.round,
        submitted_this_round=submitted_this_round,
        global_accuracy=last_round.global_accuracy if last_round else None,
    )


@router.post("/train", response_model=schemas.TrainResult)
def train(use_dp_noise: bool = False, db: Session = Depends(get_db),
          user: models.User = Depends(require_clinic_user)):
    clinic_id = _clinic_id(user)
    state = fl_core.get_or_create_model_state(db)
    weights = json.loads(state.weights_json)
    mean = json.loads(state.mean_json)
    std = json.loads(state.std_json)

    try:
        delta, local_acc, n = fl_core.train_local_from_db(db, clinic_id, weights, mean, std)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if use_dp_noise:
        delta = fl_core.add_dp_noise(delta, epsilon=2.0)

    delta_json = json.dumps(delta.tolist())
    kb_size = len(delta_json.encode()) / 1024

    target_round = state.round + 1
    existing = (db.query(models.DeltaSubmission)
                .filter(models.DeltaSubmission.clinic_id == clinic_id,
                        models.DeltaSubmission.round == target_round,
                        models.DeltaSubmission.applied == False)  # noqa: E712
                .first())
    if existing:
        existing.delta_json = delta_json
        existing.n_samples = n
        existing.local_acc = local_acc
        existing.kb_size = kb_size
    else:
        db.add(models.DeltaSubmission(
            clinic_id=clinic_id, round=target_round, delta_json=delta_json,
            n_samples=n, local_acc=local_acc, kb_size=kb_size, applied=False,
        ))
    db.commit()

    return schemas.TrainResult(
        round_trained_for=target_round, local_acc=round(local_acc, 4),
        n_samples=n, kb_transmitted=round(kb_size, 3),
        delta_norm=round(float(np.linalg.norm(delta)), 4),
    )


@router.get("/alerts", response_model=List[schemas.AlertOut])
def my_alerts(db: Session = Depends(get_db), user: models.User = Depends(require_clinic_user)):
    clinic_id = _clinic_id(user)
    own = (db.query(models.Alert)
           .filter(models.Alert.clinic_id == clinic_id)
           .order_by(models.Alert.created_at.desc()).limit(10).all())
    regional = (db.query(models.Alert)
                .filter(models.Alert.alert_type == "regional_outbreak")
                .order_by(models.Alert.created_at.desc()).limit(5).all())
    combined = sorted(own + regional, key=lambda a: a.created_at, reverse=True)
    return combined


@router.post("/predict", response_model=schemas.PredictResponse)
def predict(payload: schemas.PredictRequest, db: Session = Depends(get_db),
            user: models.User = Depends(require_clinic_user)):
    """The 'central aggregator as AI' feature: a doctor enters a patient's
    symptoms and gets a risk assessment from the GLOBAL model -- trained
    across every connected clinic's deltas, not just this clinic's own
    (usually much smaller) dataset. This is what gives a two-doctor clinic
    the same diagnostic intelligence as a hospital network.

    The response is fully explainable (exact, not approximated) because the
    model is logistic regression: every feature's contribution to the
    log-odds is reported so the doctor sees WHY, not just a number.
    """
    _clinic_id(user)  # ensures caller is a valid clinic account
    state = fl_core.get_or_create_model_state(db)
    weights = json.loads(state.weights_json)
    mean = json.loads(state.mean_json)
    std = json.loads(state.std_json)

    feature_values = payload.model_dump()
    prob, ranked = fl_core.predict_single(weights, mean, std, feature_values)

    risk_label = "High" if prob >= 0.66 else "Moderate" if prob >= 0.33 else "Low"

    top_contributions = [
        schemas.FeatureContribution(
            feature=f, label=fl_core.FEATURE_LABELS.get(f, f), value=round(float(v), 2),
            contribution=round(float(c), 4),
            direction="increases_risk" if c > 0 else "decreases_risk",
        )
        for f, v, c in ranked[:4]
    ]

    # Cross-network cohort lookup: only ever returns AGGREGATE counts/rates,
    # never another clinic's individual patient rows -- preserves the same
    # privacy invariant as the rest of the system.
    age_low, age_high = payload.age - 10, payload.age + 10
    similar = (db.query(models.Patient)
               .filter(models.Patient.age.between(age_low, age_high),
                       models.Patient.existing_condition == payload.existing_condition)
               .all())
    cohort = None
    if len(similar) >= 5:
        at_risk_n = sum(p.at_risk for p in similar)
        cohort = schemas.SimilarCohort(
            n_similar_cases=len(similar),
            at_risk_rate=round(at_risk_n / len(similar), 3),
            clinics_represented=len({p.clinic_id for p in similar}),
        )

    # Confidence: how far the prediction sits from the 50/50 decision
    # boundary. A prediction of 51% is a coin flip; 95% is decisive.
    # This is NOT the same as accuracy -- a confident model can still be
    # confidently wrong -- which is why we also surface how much data and
    # how many training rounds back the model, so a doctor can judge how
    # much to trust it.
    confidence_score = round(abs(prob - 0.5) * 2, 4)
    confidence_label = "High" if confidence_score >= 0.6 else "Moderate" if confidence_score >= 0.3 else "Low"

    total_network_patients = db.query(func.count(models.Patient.id)).scalar() or 0
    contributing_clinics = (db.query(func.count(func.distinct(models.DeltaSubmission.clinic_id)))
                             .filter(models.DeltaSubmission.applied == True).scalar() or 0)  # noqa: E712
    if state.round == 0:
        training_context = ("This model has not yet been trained on any federated round — "
                             "it's an untrained baseline. Treat this result as not meaningful yet.")
    else:
        training_context = (f"Trained over {state.round} federated round(s), with contributions from "
                             f"{contributing_clinics} clinic(s) across {total_network_patients} total "
                             f"network patient records.")

    # Rule-based differential diagnosis layer (see differential.py docstring
    # for why this is heuristic, not a trained classifier).
    symptom_flags = {
        "high_fever": payload.high_fever, "pain_behind_eyes": payload.pain_behind_eyes,
        "dry_cough": payload.dry_cough, "low_platelet": payload.low_platelet,
        "joint_bone_pain": payload.joint_bone_pain, "skin_rash": payload.skin_rash,
        "nausea_vomiting": payload.nausea_vomiting, "diarrhea_abdominal": payload.diarrhea_abdominal,
        "chills_rigors": payload.chills_rigors, "sore_throat": payload.sore_throat,
    }
    any_symptom_checked = any(symptom_flags.values())
    diff_results = differential.evaluate_differential(symptom_flags, payload.fever_days)

    if any_symptom_checked and diff_results and diff_results[0]["match_pct"] >= 15:
        top = diff_results[0]
        differential_out = [schemas.DiseaseMatch(disease=d["disease"], match_pct=d["match_pct"], reasons=d["reasons"])
                             for d in diff_results if d["match_pct"] > 0][:4]
        recommendations = top["recommendations"]
    else:
        differential_out = []
        recommendations = ["Check observed symptoms above to run a differential diagnosis suggestion."] \
            if not any_symptom_checked else \
            ["No strong match against known symptom clusters — use clinical judgment and consider broader workup."]

    # Federated context narrative: combines the recent-outbreak-alert signal
    # with the cross-network cohort lookup into one sentence, similar in
    # spirit to a "network match" summary a doctor can scan in one glance.
    since_24h = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    recent_spike_clinics = (db.query(func.count(func.distinct(models.Alert.clinic_id)))
                             .filter(models.Alert.alert_type == "local_spike",
                                     models.Alert.created_at >= since_24h).scalar() or 0)
    if cohort:
        federated_context = (
            f"{recent_spike_clinics} clinic(s) reported symptom spikes in the network in the last 24h. "
            f"Among {cohort.n_similar_cases} similar-profile cases from {cohort.clinics_represented} "
            f"clinic(s) network-wide, {cohort.at_risk_rate*100:.0f}% were flagged at-risk."
        )
    else:
        federated_context = (
            f"{recent_spike_clinics} clinic(s) reported symptom spikes in the network in the last 24h. "
            "Not enough similar-profile cases in the network yet for a cohort comparison."
        )

    return schemas.PredictResponse(
        risk_probability=round(prob, 4), risk_label=risk_label, model_round=state.round,
        top_contributions=top_contributions, cohort=cohort,
        note="Powered by the global model, trained via federated deltas from every connected clinic.",
        confidence_score=confidence_score, confidence_label=confidence_label,
        training_context=training_context,
        differential=differential_out, recommendations=recommendations,
        federated_context=federated_context,
    )


@router.get("/model_status", response_model=schemas.ModelSyncStatus)
def model_status(db: Session = Depends(get_db), user: models.User = Depends(require_clinic_user)):
    """Feature #4: lets a clinic doctor see how 'in sync' they are with the
    network -- have other clinics trained more recently than you? has the
    network grown since you last contributed?"""
    clinic_id = _clinic_id(user)
    state = fl_core.get_or_create_model_state(db)

    last_sub = (db.query(models.DeltaSubmission)
                .filter(models.DeltaSubmission.clinic_id == clinic_id, models.DeltaSubmission.applied == True)  # noqa: E712
                .order_by(models.DeltaSubmission.round.desc()).first())
    last_synced_round = last_sub.round if last_sub else None
    rounds_behind = state.round - last_synced_round if last_synced_round is not None else state.round

    since = last_sub.created_at if last_sub else datetime.datetime(1970, 1, 1)
    patients_added = (db.query(func.count(models.Patient.id))
                       .filter(models.Patient.clinic_id == clinic_id, models.Patient.created_at > since)
                       .scalar() or 0)

    total_patients = db.query(func.count(models.Patient.id)).scalar() or 0
    total_clinics = db.query(func.count(models.Clinic.id)).scalar() or 0

    return schemas.ModelSyncStatus(
        clinic_last_synced_round=last_synced_round,
        global_round=state.round,
        rounds_behind=max(0, rounds_behind),
        patients_added_since_last_sync=patients_added,
        total_network_patients=total_patients,
        total_network_clinics=total_clinics,
    )


@router.post("/treatments", response_model=schemas.TreatmentOut)
def log_treatment(payload: schemas.TreatmentCreate, db: Session = Depends(get_db),
                   user: models.User = Depends(require_clinic_user)):
    """Feature #3: doctor logs the outcome of a treatment. On its own this
    is just a log entry -- the value comes from admin.check_treatment_failures
    comparing this across every clinic to spot resistance patterns early."""
    clinic_id = _clinic_id(user)
    record = models.TreatmentRecord(clinic_id=clinic_id, **payload.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("/treatments", response_model=List[schemas.TreatmentOut])
def list_treatments(limit: int = 20, db: Session = Depends(get_db),
                     user: models.User = Depends(require_clinic_user)):
    clinic_id = _clinic_id(user)
    return (db.query(models.TreatmentRecord)
            .filter(models.TreatmentRecord.clinic_id == clinic_id)
            .order_by(models.TreatmentRecord.created_at.desc())
            .limit(limit).all())


REQUIRED_CSV_COLUMNS = ["age", "bmi", "resting_hr", "fever_days", "cough_severity",
                        "fatigue_score", "existing_condition", "bp_systolic", "at_risk"]


@router.post("/patients/bulk_import", response_model=schemas.BulkImportResult)
async def bulk_import_patients(file: UploadFile = File(...), db: Session = Depends(get_db),
                                user: models.User = Depends(require_clinic_user)):
    """Feature #5: real clinics have existing patient logs (paper or
    spreadsheet) -- typing them in one at a time doesn't scale. Expects a
    CSV with columns: age, bmi, resting_hr, fever_days, cough_severity,
    fatigue_score, existing_condition, bp_systolic, at_risk, and optionally
    symptom_tags, notes."""
    clinic_id = _clinic_id(user)
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not read file as UTF-8 text. Please upload a plain CSV.")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise HTTPException(status_code=400, detail="CSV appears to be empty")

    missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in reader.fieldnames]
    if missing_cols:
        raise HTTPException(status_code=400,
                             detail=f"CSV is missing required column(s): {', '.join(missing_cols)}")

    rows_received, rows_imported, errors = 0, 0, []
    new_patients = []
    for i, row in enumerate(reader, start=2):  # row 1 is the header
        rows_received += 1
        try:
            payload = schemas.PatientCreate(
                age=float(row["age"]), bmi=float(row["bmi"]), resting_hr=float(row["resting_hr"]),
                fever_days=float(row["fever_days"]), cough_severity=float(row["cough_severity"]),
                fatigue_score=float(row["fatigue_score"]), existing_condition=int(row["existing_condition"]),
                bp_systolic=float(row["bp_systolic"]), at_risk=int(row["at_risk"]),
                symptom_tags=row.get("symptom_tags") or None, notes=row.get("notes") or None,
            )
            new_patients.append(models.Patient(clinic_id=clinic_id, **payload.model_dump()))
            rows_imported += 1
        except Exception as e:
            errors.append(f"row {i}: {e}")

    if new_patients:
        db.add_all(new_patients)
        db.commit()
        alerts.check_local_spike(db, clinic_id)

    return schemas.BulkImportResult(
        rows_received=rows_received, rows_imported=rows_imported,
        rows_failed=rows_received - rows_imported, errors=errors[:20],
    )
