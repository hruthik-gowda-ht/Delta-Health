"""
config.py - central settings. In production, load these from environment
variables (os.environ) instead of hardcoding, especially SECRET_KEY.
"""
import os

SECRET_KEY = os.environ.get("DELTA_HEALTH_SECRET", "dev-secret-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hour shift for a clinic worker

DATABASE_URL = os.environ.get("DELTA_HEALTH_DB", "sqlite:///./delta_health.db")

# outbreak detection thresholds
OUTBREAK_SPIKE_RATIO = 2.0        # today's symptom count vs 7-day baseline average
OUTBREAK_MIN_COUNT = 3            # don't alert on tiny absolute numbers
REGIONAL_ALERT_RADIUS_KM = 10.0
REGIONAL_ALERT_MIN_CLINICS = 2    # how many nearby clinics must be spiking together

# drug efficacy / resistance detection
TREATMENT_FAILURE_MIN_RECORDS = 4      # don't flag on tiny sample sizes
TREATMENT_FAILURE_MIN_CLINICS = 2      # must be seen across multiple clinics, not one doctor's bad week
TREATMENT_FAILURE_RATE_THRESHOLD = 0.5  # fraction of "no_change"/"worsened" outcomes that triggers a flag
TREATMENT_FAILURE_WINDOW_DAYS = 14
