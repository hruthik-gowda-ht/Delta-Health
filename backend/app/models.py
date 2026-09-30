"""
models.py - the full data model.

Users -> belong to a role ('clinic' or 'admin'); clinic users are tied to
one Clinic row.

Patient rows NEVER leave this database and are NEVER included in any
DeltaSubmission payload -- only the trained weight deltas are. That's the
whole point of the system; keep that invariant when extending this file.
"""
import datetime
from sqlalchemy import (Column, Integer, String, Float, Boolean, ForeignKey,
                         DateTime, Text)
from sqlalchemy.orm import relationship
from .database import Base


class Clinic(Base):
    __tablename__ = "clinics"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    city = Column(String, nullable=False)
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    users = relationship("User", back_populates="clinic")
    patients = relationship("Patient", back_populates="clinic")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)  # "clinic" or "admin"
    full_name = Column(String, nullable=True)
    clinic_id = Column(Integer, ForeignKey("clinics.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    clinic = relationship("Clinic", back_populates="users")


class Patient(Base):
    """One row = one patient encounter entered by a clinic. This table is
    the private data referenced throughout the pitch -- it is read only by
    fl_core.train_local_from_db() for this clinic's own rows, and its
    contents are never serialized into a DeltaSubmission or any API
    response outside the owning clinic."""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    clinic_id = Column(Integer, ForeignKey("clinics.id"), nullable=False)

    age = Column(Float, nullable=False)
    bmi = Column(Float, nullable=False)
    resting_hr = Column(Float, nullable=False)
    fever_days = Column(Float, nullable=False)
    cough_severity = Column(Float, nullable=False)   # 0-3
    fatigue_score = Column(Float, nullable=False)     # 0-10
    existing_condition = Column(Integer, nullable=False)  # 0/1
    bp_systolic = Column(Float, nullable=False)

    symptom_tags = Column(String, nullable=True)   # e.g. "fever,cough,fatigue"
    at_risk = Column(Integer, nullable=False)       # doctor's assessment, 0/1 -- training label
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    clinic = relationship("Clinic", back_populates="patients")


class ModelState(Base):
    """Singleton row (id=1) holding the current global model. weights/mean/std
    are stored as JSON-encoded lists so we don't need a vector column type."""
    __tablename__ = "model_state"

    id = Column(Integer, primary_key=True, index=True)
    weights_json = Column(Text, nullable=False)
    mean_json = Column(Text, nullable=False)
    std_json = Column(Text, nullable=False)
    round = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)


class DeltaSubmission(Base):
    """A clinic's contribution for one round. delta_json is the ONLY thing
    that ever crosses from a clinic to the center -- no patient rows here."""
    __tablename__ = "delta_submissions"

    id = Column(Integer, primary_key=True, index=True)
    clinic_id = Column(Integer, ForeignKey("clinics.id"), nullable=False)
    round = Column(Integer, nullable=False)
    delta_json = Column(Text, nullable=False)
    n_samples = Column(Integer, nullable=False)
    local_acc = Column(Float, nullable=True)
    kb_size = Column(Float, nullable=False)
    applied = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class TrainingRound(Base):
    __tablename__ = "training_rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, nullable=False)
    global_accuracy = Column(Float, nullable=False)
    num_clinics = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Alert(Base):
    """Outbreak / drug-resistance style alerts. clinic_id is null for a
    regional (multi-clinic) alert."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    clinic_id = Column(Integer, ForeignKey("clinics.id"), nullable=True)
    alert_type = Column(String, nullable=False)   # "local_spike" | "regional_outbreak" | "treatment_failure"
    severity = Column(String, nullable=False)      # "info" | "warning" | "critical"
    message = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)


class TreatmentRecord(Base):
    """One row = one doctor's logged outcome after giving a treatment for a
    condition. Aggregated across clinics, this is how the network notices
    'this antibiotic stopped working' faster than any single clinic could."""
    __tablename__ = "treatment_records"

    id = Column(Integer, primary_key=True, index=True)
    clinic_id = Column(Integer, ForeignKey("clinics.id"), nullable=False)
    treatment_name = Column(String, nullable=False)
    condition_tag = Column(String, nullable=False)   # e.g. "bacterial_fever", "uti", "resp_infection"
    outcome = Column(String, nullable=False)          # "improved" | "no_change" | "worsened"
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
