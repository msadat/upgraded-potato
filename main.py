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
from typing import Any, Dict, List, Optional, Set

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from bearing_capacity_engine import calculate_bearing_capacity


APP_TITLE = "GeoTechHub Bearing Capacity API"
APP_VERSION = "1.0.2"


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "Shallow foundation bearing capacity, eccentric loading, contact pressure, "
        "sliding, overturning, settlement, and PDF report API."
    ),
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
    """
    Optional API key protection.

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
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


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

        safe_name = (
            result.get("project_name", "bearing-capacity-report")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

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
    """
    Create a calculation PDF using ReportLab and return a temporary file path.
    Equations are rendered as LaTeX-style images using Matplotlib mathtext.
    """

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
    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    small_style = ParagraphStyle(
        "GeoTechHubSmall",
        parent=normal_style,
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        spaceAfter=4,
    )

    story: List[Any] = []

    story.append(Paragraph("GeoTechHub Bearing Capacity Design Summary", title_style))
    story.append(Paragraph(f"Project: {safe_text(result.get('project_name', ''))}", normal_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 0.15 * inch))

    pass_fail = result.get("pass_fail", {})
    overall_status = pass_fail.get("overall_status", "UNKNOWN")
    governing_check = pass_fail.get("governing_check", "Not available")

    if overall_status == "PASS":
        status_color = colors.HexColor("#107c41")
    elif overall_status == "FAIL":
        status_color = colors.HexColor("#b42318")
    else:
        status_color = colors.HexColor("#b54708")

    status_table = Table(
        [
            ["Overall Status", overall_status],
            ["Governing Check", safe_text(governing_check)],
        ],
        colWidths=[1.65 * inch, 5.2 * inch],
    )

    status_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f4f7")),
                ("TEXTCOLOR", (1, 0), (1, 0), status_color),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )

    story.append(status_table)
    story.append(Spacer(1, 0.2 * inch))

    add_section_table(
        story,
        "Input Summary",
        flatten_for_table(result.get("input_summary", {})),
        heading_style,
    )

    add_section_table(
        story,
        "Effective Footing",
        flatten_for_table(result.get("effective_footing", {})),
        heading_style,
    )

    add_section_table(
        story,
        "Bearing Capacity",
        flatten_for_table(
            result.get("bearing_capacity", {}),
            exclude={"contact_pressure_reference"},
        ),
        heading_style,
    )

    add_section_table(
        story,
        "Contact Pressure",
        flatten_for_table(
            result.get("contact_pressure", {}),
            exclude={"corners"},
        ),
        heading_style,
    )

    add_section_table(
        story,
        "Sliding Check",
        flatten_for_table(result.get("sliding", {})),
        heading_style,
    )

    add_section_table(
        story,
        "Overturning Check",
        flatten_for_table(result.get("overturning", {})),
        heading_style,
    )

    add_section_table(
        story,
        "Settlement Estimate",
        flatten_for_table(result.get("settlement", {})),
        heading_style,
    )

    story.append(Paragraph("Engineering Assumptions", heading_style))

    assumptions = result.get("engineering_assumptions", [])
    if assumptions:
        for item in assumptions:
            story.append(Paragraph(f"• {safe_text(item)}", small_style))
    else:
        story.append(Paragraph("No engineering assumptions were returned by the calculation engine.", small_style))

    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Formula Sheet", heading_style))
    story.append(
        Paragraph(
            "The following governing equations are provided for calculation transparency.",
            small_style,
        )
    )
    story.append(Spacer(1, 0.05 * inch))

    for formula_item in professional_pdf_formulas():
        story.append(Paragraph(f"<b>{safe_text(formula_item['title'])}</b>", small_style))

        equation_img = render_latex_equation_image(formula_item["equation"])
        story.append(equation_img)

        if formula_item.get("note"):
            story.append(Paragraph(safe_text(formula_item["note"]), small_style))

        story.append(Spacer(1, 0.12 * inch))

    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            "Disclaimer: This report is generated for preliminary engineering screening only. "
            "It does not replace the project geotechnical report, applicable codes, "
            "site-specific engineering evaluation, or professional engineering judgment.",
            small_style,
        )
    )

    doc.build(story)

    return tmp.name


def add_section_table(
    story: List[Any],
    title: str,
    rows: List[List[Any]],
    heading_style: Any,
) -> None:
    story.append(Paragraph(title, heading_style))

    table_rows: List[List[Any]] = [["Parameter", "Value"]]

    if rows:
        for parameter, value in rows:
            table_rows.append(
                [
                    Paragraph(safe_text(parameter), get_pdf_small_style()),
                    Paragraph(safe_text(value), get_pdf_small_style()),
                ]
            )
    else:
        table_rows.append(
            [
                Paragraph("No data", get_pdf_small_style()),
                Paragraph("Not available", get_pdf_small_style()),
            ]
        )

    table = Table(
        table_rows,
        colWidths=[3.0 * inch, 3.85 * inch],
        repeatRows=1,
    )

    table.setStyle(default_table_style())

    story.append(table)
    story.append(Spacer(1, 0.12 * inch))


def get_pdf_small_style() -> ParagraphStyle:
    styles = getSampleStyleSheet()

    return ParagraphStyle(
        "GeoTechHubTableSmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
    )


def default_table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#f5f7fa")],
            ),
        ]
    )


def professional_pdf_formulas() -> List[Dict[str, str]]:
    """
    Professional formula sheet for PDF report.
    Equations use Matplotlib mathtext syntax.
    """
    return [
        {
            "title": "General Bearing Capacity Equation",
            "equation": (
                r"$q_{ult}=cN_c s_c d_c i_c+qN_q s_q d_q i_q+"
                r"\frac{1}{2}\gamma B N_{\gamma}s_{\gamma}d_{\gamma}i_{\gamma}$"
            ),
            "note": "General form used by Meyerhof, Hansen, and Vesic-type bearing capacity methods.",
        },
        {
            "title": "Foundation-Level Surcharge",
            "equation": r"$q=\gamma D_f$",
            "note": "Surcharge pressure at the foundation bearing elevation.",
        },
        {
            "title": "Bearing Capacity Factor Nq",
            "equation": r"$N_q=e^{\pi\tan\phi}\tan^2\left(45^\circ+\frac{\phi}{2}\right)$",
            "note": "Used for drained bearing capacity where soil friction angle is considered.",
        },
        {
            "title": "Bearing Capacity Factor Nc",
            "equation": r"$N_c=\frac{N_q-1}{\tan\phi}$",
            "note": "For phi equal to zero, the classical limiting value Nc = 5.14 is commonly used.",
        },
        {
            "title": "Bearing Capacity Factor Ngamma",
            "equation": r"$N_{\gamma}\approx2\left(N_q+1\right)\tan\phi$",
            "note": "Approximate expression used in simplified bearing capacity calculations.",
        },
        {
            "title": "Biaxial Eccentricity",
            "equation": r"$e_x=\frac{M_y}{P},\qquad e_y=\frac{M_x}{P}$",
            "note": "Moments are converted into load eccentricities relative to the footing centroid.",
        },
        {
            "title": "Effective Footing Dimensions",
            "equation": r"$B'=B-2e_x,\qquad L'=L-2e_y,\qquad A'=B'L'$",
            "note": "Effective area method for eccentric vertical loading.",
        },
        {
            "title": "Effective Bearing Pressure",
            "equation": r"$q_{eff}=\frac{P}{A'}$",
            "note": "Applied bearing pressure based on reduced effective footing area.",
        },
        {
            "title": "Gross and Net Allowable Bearing",
            "equation": (
                r"$q_{allow,gross}=\frac{q_{ult,gross}}{FS},\qquad "
                r"q_{allow,net}=\frac{q_{ult,net}}{FS}$"
            ),
            "note": "Gross bearing includes surcharge; net bearing subtracts foundation-level surcharge.",
        },
        {
            "title": "Linear Contact Pressure under Biaxial Moment",
            "equation": r"$q(x,y)=\frac{P}{A}\pm\frac{M_x y}{I_x}\pm\frac{M_y x}{I_y}$",
            "note": "Used to estimate corner pressures and identify uplift or partial bearing.",
        },
        {
            "title": "Rectangular Footing Section Properties",
            "equation": r"$A=BL,\qquad I_x=\frac{BL^3}{12},\qquad I_y=\frac{LB^3}{12}$",
            "note": "Plan dimensions used for linear contact pressure distribution.",
        },
        {
            "title": "Sliding Factor of Safety",
            "equation": r"$FS_{sliding}=\frac{P\tan\delta+c_aA+P_p}{H}$",
            "note": "Passive resistance should only be included when justified by site conditions.",
        },
        {
            "title": "Overturning Factor of Safety",
            "equation": r"$FS_{OT}=\frac{M_{resisting}}{M_{overturning}}$",
            "note": "Calculated about the principal footing axes.",
        },
        {
            "title": "Elastic Settlement Estimate",
            "equation": r"$S=\frac{qB(1-\nu^2)I_s}{E_s}$",
            "note": "Preliminary immediate settlement estimate for service-level screening.",
        },
    ]


def render_latex_equation_image(equation: str) -> Image:
    """
    Render a LaTeX-style equation using Matplotlib mathtext and return
    a ReportLab Image flowable for insertion into the PDF.
    """

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()

    fig = plt.figure(figsize=(6.4, 0.65), dpi=220)
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.01,
        0.5,
        equation,
        fontsize=11,
        va="center",
        ha="left",
        color="black",
    )

    fig.savefig(
        tmp.name,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=False,
        facecolor="white",
    )

    plt.close(fig)

    img = Image(tmp.name)

    max_width = 6.4 * inch

    if img.drawWidth > max_width:
        scale = max_width / img.drawWidth
        img.drawWidth *= scale
        img.drawHeight *= scale

    return img


def flatten_for_table(
    data: Dict[str, Any],
    exclude: Optional[Set[str]] = None,
) -> List[List[Any]]:
    exclude = exclude or set()
    rows: List[List[Any]] = []

    if not isinstance(data, dict):
        return rows

    for key, value in data.items():
        if key in exclude:
            continue

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (dict, list)):
                    continue

                rows.append(
                    [
                        pretty_label(f"{key}.{sub_key}"),
                        format_value(sub_value),
                    ]
                )

        elif isinstance(value, list):
            continue

        else:
            rows.append(
                [
                    pretty_label(key),
                    format_value(value),
                ]
            )

    return rows


def pretty_label(value: str) -> str:
    value = value.replace("_", " ").replace(".", " - ")
    return value.title()


def format_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, float):
        return f"{value:,.4g}"

    if isinstance(value, int):
        return f"{value:,}"

    return str(value)


def safe_text(value: Any) -> str:
    """
    ReportLab Paragraph uses a small XML-like markup language.
    Escape special characters to avoid PDF rendering errors.
    """
    text = str(value)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
