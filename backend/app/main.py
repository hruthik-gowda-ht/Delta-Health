from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

from .database import Base, engine
from .routers import auth, clinic, admin, public

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Delta Network - Federated Healthcare AI")

BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.include_router(auth.router)
app.include_router(clinic.router)
app.include_router(admin.router)
app.include_router(public.router)


@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse(url="/login")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/clinic/dashboard", response_class=HTMLResponse)
def clinic_dashboard_page(request: Request):
    return templates.TemplateResponse("clinic_dashboard.html", {"request": request})


@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard_page(request: Request):
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})


@app.get("/admin/outbreak-monitor", response_class=HTMLResponse)
def outbreak_monitor_page(request: Request):
    return templates.TemplateResponse("outbreak_monitor.html", {"request": request})


@app.get("/admin/drug-efficacy", response_class=HTMLResponse)
def drug_efficacy_page(request: Request):
    return templates.TemplateResponse("drug_efficacy.html", {"request": request})


@app.get("/admin/stock-advisory", response_class=HTMLResponse)
def stock_advisory_page(request: Request):
    return templates.TemplateResponse("stock_advisory.html", {"request": request})


@app.get("/public", response_class=HTMLResponse)
def public_dashboard_page(request: Request):
    return templates.TemplateResponse("public_dashboard.html", {"request": request})
