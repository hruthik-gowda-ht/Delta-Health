# Delta Network — Privacy-First Federated Healthcare AI

A production-structured full-stack app: FastAPI backend, SQLite/Postgres-ready
database, JWT auth, two role-based portals (clinic + public health), and the
actual federated learning delta algorithm wired end to end.

## Architecture

```
delta-health/
└── backend/
    ├── requirements.txt
    ├── seed.py                  # creates demo clinics, logins, patients
    └── app/
        ├── main.py              # FastAPI app, routes, template pages
        ├── config.py            # settings (secret key, thresholds)
        ├── database.py          # SQLAlchemy engine/session
        ├── models.py            # ORM: Clinic, User, Patient, ModelState,
        │                        #      DeltaSubmission, TrainingRound, Alert
        ├── schemas.py            # Pydantic request/response models
        ├── security.py          # password hashing (PBKDF2) + JWT
        ├── deps.py               # auth dependencies (role-gated)
        ├── fl_core.py            # the delta algorithm itself
        ├── alerts.py             # outbreak detection (local + regional)
        ├── routers/
        │   ├── auth.py           # POST /auth/login
        │   ├── clinic.py         # clinic portal API
        │   └── admin.py          # public health portal API
        ├── templates/
        │   ├── login.html
        │   ├── clinic_dashboard.html
        │   └── admin_dashboard.html
        └── static/
```

## Why this structure (for judges)

- **Two real, separately-authenticated portals**, not a toggle in one UI:
  `/login` issues a JWT with a `role` claim; `require_clinic_user` /
  `require_admin_user` FastAPI dependencies enforce it server-side on every
  endpoint, not just hidden in the frontend.
- **Patient data lives in one table, `patients`, scoped by `clinic_id`.**
  Every query that reads it is filtered to the requesting clinic's own id
  (see `_clinic_id()` guard in `routers/clinic.py`). No endpoint anywhere
  returns another clinic's patient rows.
- **`DeltaSubmission` is the only artifact that crosses the clinic → admin
  boundary.** It stores `delta_json` (a plain vector) and summary stats —
  never a foreign key into `patients`, never per-patient values.
- **`fl_core.py` is the whole algorithm**, deliberately separated from the
  web/DB plumbing so you can explain it on a whiteboard in 30 seconds:
  train locally → subtract old weights = delta → clip outliers → weighted
  average → evaluate on a shared benchmark set.

## Setup

```bash
cd backend
python3 -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt

python3 seed.py          # creates delta_health.db with demo clinics + users
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` — it redirects to `/login`.

**Demo logins** (created by `seed.py`):

| Portal | Username | Password |
|---|---|---|
| Clinic 1 — Sunrise Family Clinic | `clinic1` | `clinic123` |
| Clinic 2 — Green Cross Neighborhood Clinic | `clinic2` | `clinic123` |
| Clinic 3 — Dharwad Community Health Center | `clinic3` | `clinic123` |
| Clinic 4 — Vidyanagar Kirana-Health Post | `clinic4` | `clinic123` |
| Public health officer | `healthofficer` | `admin123` |

## Demo script (what to click, in order)

1. **Log in as `clinic1`.** Show the stat cards (empty local accuracy — no
   model trained yet). Add 2-3 patient records through the form — point out
   this data is now only in this clinic's row of the local `patients` table.
2. **Click "Train locally & submit delta."** Show the result panel: local
   accuracy, `n_samples` used, and — the key visual — **KB transmitted**
   (a few hundred bytes) vs. what a raw patient export would weigh.
3. Repeat step 1–2 quickly for `clinic2`, `clinic3`, `clinic4` (open in
   separate incognito windows or just log out/in — that's a great visual
   of "these are different physical locations").
4. **Log in as `healthofficer`.** Show the Overview: pending submissions
   count should now be 4. Click **"Aggregate pending deltas"** — the chart
   updates live with the new global accuracy, and the round counter
   advances. This is the FedAvg step happening for real, in front of them.
5. **Trigger an outbreak alert**: as a clinic user, add several patients
   with `fever_days >= 1` in one sitting (5+ quickly) — this fires
   `check_local_spike` automatically after each patient save. Switch to
   the admin portal and click **"Check regional outbreak signal"** — if
   two nearby clinics both spiked, you get a `regional_outbreak` critical
   alert citing the actual clinic names and distance, which is the "45
   clinics in a 10km radius" feature from the pitch, working on real
   haversine distance between clinic lat/lng.
6. **Risk heatmap** on the admin portal shows at-risk rate and alert count
   per clinic with color-coded bars — this is your "epidemiological
   heatmap without touching a patient record" story.

## What's real vs. what's simplified (say this proactively if asked)

**Real and working:**
- Actual federated averaging (FedAvg) over real per-clinic logistic
  regression training, computed from database rows
- Real JWT-based auth with server-enforced roles, not just a UI switch
- Real outbreak detection using clinic-relative baselines and haversine
  distance between real clinic coordinates
- Real privacy accounting — the dashboard shows literal KB transmitted per
  delta submission, pulled from the actual serialized payload size

**Simplified for a 2-day build (be upfront about this):**
- Model is logistic regression on 8 tabular features, not a deep model —
  the algorithm generalizes to any gradient-trainable model, this keeps
  training near-instant for a live demo
- No differential-privacy accounting library — `add_dp_noise` adds
  calibrated Gaussian noise as a stand-in for a proper DP mechanism
  (e.g. Opacus, or the Gaussian mechanism with a real privacy budget)
- No secure aggregation (cryptographic protocol so the server can't see
  any individual clinic's delta, only the sum) — worth naming as the next
  hardening step if asked
- SQLite for the demo; `DATABASE_URL` in `config.py` swaps to Postgres
  with zero code changes elsewhere (SQLAlchemy handles both)
- Patients are simulated demo data seeded by `seed.py`; the schema is
  designed to accept real de-identified clinic exports with the same
  column names

## Extending toward a real deployment

- Swap `add_dp_noise` for a proper (ε, δ)-differential-privacy library
- Add secure aggregation (e.g. via a simple secret-sharing scheme) so the
  aggregator only ever sees the *sum* of deltas
- Add an audit log table recording every access to `/clinic/patients`
  (required for a real HIPAA-adjacent deployment)
- Move `SECRET_KEY` to an environment variable / secrets manager (already
  wired to read `DELTA_HEALTH_SECRET` env var, just needs to be set)
- Add clinic self-registration with admin approval instead of `seed.py`
