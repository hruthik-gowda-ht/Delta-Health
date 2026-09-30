from pydantic import BaseModel, Field
from typing import Optional, List
import datetime


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    clinic_id: Optional[int] = None
    full_name: Optional[str] = None


class PatientCreate(BaseModel):
    age: float = Field(..., ge=0, le=120)
    bmi: float = Field(..., ge=10, le=70)
    resting_hr: float = Field(..., ge=30, le=220)
    fever_days: float = Field(..., ge=0, le=30)
    cough_severity: float = Field(..., ge=0, le=3)
    fatigue_score: float = Field(..., ge=0, le=10)
    existing_condition: int = Field(..., ge=0, le=1)
    bp_systolic: float = Field(..., ge=60, le=260)
    symptom_tags: Optional[str] = None
    at_risk: int = Field(..., ge=0, le=1)
    notes: Optional[str] = None


class PatientOut(PatientCreate):
    id: int
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class ClinicSummary(BaseModel):
    clinic_id: int
    clinic_name: str
    patient_count: int
    at_risk_rate: float
    last_local_acc: Optional[float]
    global_round: int
    submitted_this_round: bool
    global_accuracy: Optional[float]


class TrainResult(BaseModel):
    round_trained_for: int
    local_acc: float
    n_samples: int
    kb_transmitted: float
    delta_norm: float


class AlertOut(BaseModel):
    id: int
    clinic_id: Optional[int]
    alert_type: str
    severity: str
    message: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class ClinicHeatmapPoint(BaseModel):
    clinic_id: int
    name: str
    city: str
    lat: float
    lng: float
    patient_count: int
    at_risk_rate: float
    alert_count_24h: int


class AdminOverview(BaseModel):
    current_round: int
    global_accuracy: Optional[float]
    total_clinics: int
    total_patients: int
    pending_submissions: int
    accuracy_history: List[dict]


class PredictRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    bmi: float = Field(..., ge=10, le=70)
    resting_hr: float = Field(..., ge=30, le=220)
    fever_days: float = Field(..., ge=0, le=30)
    cough_severity: float = Field(..., ge=0, le=3)
    fatigue_score: float = Field(..., ge=0, le=10)
    existing_condition: int = Field(..., ge=0, le=1)
    bp_systolic: float = Field(..., ge=60, le=260)
    # optional observed-symptom checklist, used only for the rule-based
    # differential diagnosis layer (see differential.py) -- NOT fed into
    # the real ML risk model, which uses the numeric fields above.
    high_fever: bool = False
    pain_behind_eyes: bool = False
    dry_cough: bool = False
    low_platelet: bool = False
    joint_bone_pain: bool = False
    skin_rash: bool = False
    nausea_vomiting: bool = False
    diarrhea_abdominal: bool = False
    chills_rigors: bool = False
    sore_throat: bool = False


class DiseaseMatch(BaseModel):
    disease: str
    match_pct: float
    reasons: List[str]


class FeatureContribution(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float
    direction: str  # "increases_risk" | "decreases_risk"


class SimilarCohort(BaseModel):
    n_similar_cases: int
    at_risk_rate: float
    clinics_represented: int


class PredictResponse(BaseModel):
    risk_probability: float
    risk_label: str  # "Low" | "Moderate" | "High"
    model_round: int
    top_contributions: List[FeatureContribution]
    cohort: Optional[SimilarCohort]
    note: str
    confidence_score: float
    confidence_label: str  # "Low" | "Moderate" | "High"
    training_context: str
    differential: List[DiseaseMatch]
    recommendations: List[str]
    federated_context: str


class TreatmentCreate(BaseModel):
    treatment_name: str = Field(..., min_length=1, max_length=120)
    condition_tag: str = Field(..., min_length=1, max_length=120)
    outcome: str = Field(..., pattern="^(improved|no_change|worsened)$")
    notes: Optional[str] = None


class TreatmentOut(TreatmentCreate):
    id: int
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class TreatmentEfficacyRow(BaseModel):
    treatment_name: str
    condition_tag: str
    total_records: int
    clinics_represented: int
    failure_rate: float


class ModelSyncStatus(BaseModel):
    clinic_last_synced_round: Optional[int]
    global_round: int
    rounds_behind: int
    patients_added_since_last_sync: int
    total_network_patients: int
    total_network_clinics: int


class BulkImportResult(BaseModel):
    rows_received: int
    rows_imported: int
    rows_failed: int
    errors: List[str]


class TreatmentDiagnostic(BaseModel):
    treatment: str
    condition: str
    records: int
    clinics: int
    failure_rate: float


class TreatmentCheckResult(BaseModel):
    alerts_created: List[str]
    diagnostics: List[TreatmentDiagnostic]
    min_records: int
    min_clinics: int
    rate_threshold: float


class SpikingClinic(BaseModel):
    name: str
    city: str


class RegionalCheckResult(BaseModel):
    alert_created: Optional[str]
    spiking_clinics: List[SpikingClinic]
    largest_cluster_size: int
    min_clinics_required: int
    radius_km: float


class ClinicTimelineSeries(BaseModel):
    clinic_id: int
    name: str
    city: str
    lat: float
    lng: float
    is_spiking: bool
    daily_counts: List[int]  # aligned with OutbreakTimelineResponse.dates


class OutbreakTimelineResponse(BaseModel):
    dates: List[str]
    series: List[ClinicTimelineSeries]
    radius_km: float
    min_clinics_required: int


class TreatmentTimelineSeries(BaseModel):
    treatment: str
    condition: str
    total_records: int
    failure_rate: float
    daily_improved: List[int]
    daily_no_change: List[int]
    daily_worsened: List[int]


class DrugEfficacyTimelineResponse(BaseModel):
    dates: List[str]
    series: List[TreatmentTimelineSeries]


class ClinicStockAdvisory(BaseModel):
    clinic_id: int
    clinic_name: str
    city: str
    likely_pattern: str
    reason: str
    recommended_supplies: List[str]


class DemandTrend(BaseModel):
    treatment_name: str
    this_week_count: int
    previous_week_count: int
    change_pct: float


class StockAdvisoryResponse(BaseModel):
    clinic_advisories: List[ClinicStockAdvisory]
    demand_trends: List[DemandTrend]
