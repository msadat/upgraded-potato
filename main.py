"""
GeoTechHub Bearing Capacity FastAPI Backend
Render-ready API for WordPress/JavaScript frontend integration.

Run locally:
    uvicorn main:app --reload

Render start command:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bearing_capacity_engine import BearingCapacityInput, calculate_bearing_capacity


APP_TITLE = "GeoTechHub Bearing Capacity API"
APP_VERSION = "1.0.0"

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Shallow foundation bearing capacity, eccentric loading, contact pressure, sliding, overturning, settlement, and PDF report API.",
)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BearingCapacityRequest(BaseModel):
    project_name: str = Field("GeoTechHub Bearing Capacity Check")
    method: str = Field("Meyerhof", description="Terzaghi, Meyerhof, Hansen, or Vesic")
    footing_shape: str = Field("rectangular", description="strip, square, circular, or rectangular")

    B_ft: float = Field(6.0, gt=0)
    L_ft: float = Field(10.0, gt=0)
    Df_ft: float = Field(3.0, ge=0)

    P_kip: float = Field(250.0, gt=0)
    Mx_kipft: float = 0.0
    My_kipft: float = 0.0
    Hx_kip: float = 0.0
    Hy_kip: float = 0.0

    cohesion_psf: float = Field(500.0, ge=0)
    phi_deg: float = Field(30.0, ge=0, lt=50)
    gamma_pcf: float = Field(120.0, gt=0)
    gamma_sat_pcf: float = Field(125.0, gt=0)
    water_table_depth_ft: float = Field(999.0, ge=0)

    FS_bearing_required: float = Field(3.0, gt=0)
    FS_sliding_required: float = Field(1.5, gt=0)
    FS_overturning_required: float = Field(2.0, gt=0)

    base_friction_angle_deg: float = Field(25.0, ge=0, lt=50)
    base_adhesion_psf: float = Field(0.0, ge=0)
    passive_resistance_kip: float = Field(0.0, ge=0)
    include_passive_resistance: bool = False

    elastic_modulus_ksf: float = Field(500.0, gt=0)
    poisson_ratio: float = Field(0.35, ge=0, lt=0.5)
    settlement_influence_factor: float = Field(1.0, gt=0)
    allowable_settlement_in: float = Field(1.0, gt=0)


class MethodsResponse(BaseModel):
    methods: List[str]
    footing_shapes: List[str]


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Optional API key protection.

    Set GEOTECHHUB_API_KEY in Render environment variables to require x-api-key.
    If GEOTECHHUB_API_KEY is not set, the API remains open for testing.
    """
    required_key = os.getenv("GEOTECHHUB_API_KEY")
    if required_key and x_api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": APP_TITLE,
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.get("/bearing-capacity/methods", response_model=MethodsResponse)
def methods() -> MethodsResponse:
    return MethodsResponse(
        methods=["Terzaghi", "Meyerhof", "Hansen", "Vesic"],
        footing_shapes=["strip", "square", "circular", "rectangular"],
    )


@app.post("/bearing-capacity")
def bearing_capacity_endpoint(
    payload: BearingCapacityRequest,
    _: None = Depends(require_api_key),
) -> Dict[str, Any]:
    try:
        data = payload.model_dump()
        return calculate_bearing_capacity(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Calculation failed: {exc}") from exc


@app.post("/bearing-capacity/pdf")
def bearing_capacity_pdf_endpoint(
    payload: BearingCapacityRequest,
    _: None = Depends(require_api_key),
) -> FileResponse:
    try:
        result = calculate_bearing_capacity(payload.model_dump())
        pdf_path = build_pdf_report(result)
        safe_name = result.get("project_name", "bearing-capacity-report").replace(" ", "_")
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"{safe_name}.pdf",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}") from exc


def build_pdf_report(result: Dict[str, Any]) -> str:
    """Create a calculation PDF using ReportLab and return temp file path."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()

    doc = SimpleDocTemplate(
        tmp.name,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2 = styles["Heading2"]
    normal = styles["BodyText"]
    small = ParagraphStyle("small", parent=normal, fontSize=8, leading=10)

    story: List[Any] = []
    story.append(Paragraph("GeoTechHub Bearing Capacity Design Summary", title_style))
    story.append(Paragraph(f"Project: {result.get('project_name', '')}", normal))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal))
    story.append(Spacer(1, 0.15 * inch))

    pf = result["pass_fail"]
    status_color = colors.green if pf["overall_status"] == "PASS" else colors.red if pf["overall_status"] == "FAIL" else colors.orange
    status_table = Table([
        ["Overall Status", pf["overall_status"]],
        ["Governing Check", pf["governing_check"]],
    ], colWidths=[1.65 * inch, 5.2 * inch])
    status_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("TEXTCOLOR", (1, 0), (1, 0), status_color),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(status_table)
    story.append(Spacer(1, 0.2 * inch))

    add_section_table(story, "Input Summary", flatten_for_table(result["input_summary"]), h2)
    add_section_table(story, "Effective Footing", flatten_for_table(result["effective_footing"]), h2)
    add_section_table(story, "Bearing Capacity", flatten_for_table(result["bearing_capacity"], exclude={"contact_pressure_reference"}), h2)
    add_section_table(story, "Contact Pressure", flatten_for_table(result["contact_pressure"], exclude={"corners"}), h2)
    add_section_table(story, "Sliding Check", flatten_for_table(result["sliding"]), h2)
    add_section_table(story, "Overturning Check", flatten_for_table(result["overturning"]), h2)
    add_section_table(story, "Settlement Estimate", flatten_for_table(result["settlement"]), h2)

    story.append(Paragraph("Engineering Assumptions", h2))
    for item in result["engineering_assumptions"]:
        story.append(Paragraph(f"• {item}", small))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Formula Sheet", h2))
    formula_rows = [[f["name"], f["formula"]] for f in result["formula_sheet"]]
    formula_table = Table([["Item", "Formula"]] + formula_rows, colWidths=[2.0 * inch, 4.85 * inch])
    formula_table.setStyle(default_table_style())
    story.append(formula_table)

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Disclaimer: This report is generated for preliminary engineering screening and does not replace the project geotechnical report, applicable codes, or professional engineering judgment.",
        small,
    ))

    doc.build(story)
    return tmp.name


def add_section_table(story: List[Any], title: str, rows: List[List[Any]], h2_style: Any) -> None:
    story.append(Paragraph(title, h2_style))
    table = Table([["Parameter", "Value"]] + rows, colWidths=[3.0 * inch, 3.85 * inch])
    table.setStyle(default_table_style())
    story.append(table)
    story.append(Spacer(1, 0.12 * inch))


def default_table_style() -> TableStyle:
    from reportlab.lib import colors

    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
    ])


def flatten_for_table(data: Dict[str, Any], exclude: Optional[set] = None) -> List[List[Any]]:
    exclude = exclude or set()
    rows: List[List[Any]] = []
    for k, v in data.items():
        if k in exclude:
            continue
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, (dict, list)):
                    continue
                rows.append([f"{k}.{sub_k}", str(sub_v)])
        elif isinstance(v, list):
            continue
        else:
            rows.append([k, str(v)])
    return rows
