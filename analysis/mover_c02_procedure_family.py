#!/usr/bin/env python3
"""Outcome-blind broad procedure-family mapping for MOVER C02.

This intentionally uses transparent keyword rules rather than an outcome-tuned
classifier.  It is a coarse context sensitivity, not a clinical coding system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from c02_runtime import require_private_path, secure_directory


RULES: list[tuple[str, tuple[str, ...]]] = [
    ("transplant", ("TRANSPLANT",)),
    (
        "cardiac",
        (
            "CORONARY ARTERY BYPASS", "CABG", "CARDIAC", "HEART", "VALVE",
            "VENTRICULAR ASSIST", "PERICARD", "MAZE PROCEDURE", "SEPTAL DEFECT",
        ),
    ),
    (
        "thoracic",
        (
            "THORAC", "LOBECTOM", "PNEUMONECTOM", "LUNG", "PLEUR", "MEDIASTIN",
            "BRONCHOSCOP", "ESOPHAGECTOM",
        ),
    ),
    (
        "neuro_spine",
        (
            "CRANI", "BRAIN", "CEREBR", "SPINE", "SPINAL", "LAMINECTOM",
            "LAMINOPLAST", "DISCECTOM", "DEEP BRAIN STIMULATION", "SHUNT",
            "NEURO", "INTRACRANIAL", "MENINGIOMA",
        ),
    ),
    (
        "vascular",
        (
            "ENDARTERECTOM", "VASCULAR", "ARTERIAL", "ARTERY", "AORT",
            "ANGIOPLAST", "ANGIOGRAM", "EMBOLECTOM", "THROMBECTOM", "FISTULA",
            "VASCULAR BYPASS", "VENOUS", "VEIN", "VARICOSE",
        ),
    ),
    (
        "orthopaedic",
        (
            "ARTHROPLAST", "ARTHROSCOP", "ORIF", "FRACTURE", "ORTHOP",
            "HIP", "KNEE", "SHOULDER", "ELBOW", "FEMUR", "TIBIA", "FIBULA",
            "ANKLE", "FOOT", "HAND", "WRIST", "TENDON", "LIGAMENT", "BONE",
            "AMPUTATION", "OSTEOTOM", "MENISC", "CARPAL TUNNEL",
        ),
    ),
    (
        "urology",
        (
            "CYSTOSCOP", "URETER", "TURBT", "TURP", "NEPHR", "PROSTAT",
            "BLADDER", "URINARY", "UROLOG", "LITHOTRIPS", "PENILE", "TESTIC",
            "SCROT", "KIDNEY", "PYELOGRAM",
        ),
    ),
    (
        "gynaecology_obstetric",
        (
            "HYSTERECT", "SALPING", "OOPHORECT", "OVARI", "UTER", "VAGIN",
            "VULV", "CERVIX", "CESAREAN", "DILATION AND CURETTAGE", "GYNECOL",
            "PELVIC EXENTERATION",
        ),
    ),
    (
        "gastrointestinal_hpb",
        (
            "LAPAROTOM", "LAPAROSCOP", "CHOLECYST", "COLECTOM", "BOWEL",
            "COLON", "RECTAL", "ANORECTAL", "ILEOSTOM", "COLOSTOM", "HERNIA",
            "GASTR", "PANCREA", "HEPAT", "LIVER", "BILIARY", "APPENDECT",
            "WHIPPLE", "ABDOM", "SMALL INTESTINE", "PROCTECTOM", "HEMORRHOID",
        ),
    ),
    (
        "ent_head_neck",
        (
            "TONSIL", "ADENOID", "SINUS", "NASAL", "SEPTOPLAST", "LARYNG",
            "PHARYNG", "TRACHEOST", "THYROID", "PARATHYROID", "PAROTID", "NECK",
            "MANDIB", "MAXILL", "ORAL", "TONGUE", "EAR", "MASTOID",
        ),
    ),
    (
        "breast_plastic_wound",
        (
            "BREAST", "MASTECTOM", "MAMMOPLAST", "FLAP", "SKIN GRAFT",
            "DEBRIDEMENT", "WOUND", "RECONSTRUCTION", "PLASTIC", "BURN",
        ),
    ),
    (
        "ophthalmic",
        ("CATARACT", "VITRECTOM", "RETINA", "CORNEA", "EYE", "OPHTHALM"),
    ),
    (
        "endoscopy_interventional",
        (
            "ENDOSCOP", "COLONOSCOP", "EGD", "ERCP", "INTERVENTIONAL RADIOLOGY",
            "IMAGE-GUIDED", "BIOPSY", "CATHETER INSERTION", "PORT INSERTION",
        ),
    ),
]


def procedure_family(value: object) -> str:
    text = "" if value is None else str(value).upper().strip()
    if not text or text == "<MISSING>" or text == "NAN":
        return "missing"
    for family, keywords in RULES:
        if any(keyword in text for keyword in keywords):
            return family
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.cohort = require_private_path(args.cohort)
    args.output_dir = secure_directory(args.output_dir)
    # Deliberately do not read target or MAP columns in this audit.
    d = pd.read_csv(
        args.cohort,
        usecols=["procedure_common", "prior_procedure_common", "interval_days"],
        low_memory=False,
    )
    d["current_family"] = d["procedure_common"].map(procedure_family)
    d["prior_family"] = d["prior_procedure_common"].map(procedure_family)
    d["same_exact_procedure"] = d["procedure_common"].eq(d["prior_procedure_common"])
    d["same_broad_family"] = d["current_family"].eq(d["prior_family"])
    counts = (
        d.groupby(["current_family", "prior_family"], observed=True)
        .size()
        .rename("pairs")
        .reset_index()
        .sort_values("pairs", ascending=False)
    )
    counts.to_csv(args.output_dir / "outcome_blind_family_pair_counts.csv", index=False)
    summary = {
        "status": "OUTCOME_BLIND_PROCEDURE_MAPPING_FROZEN",
        "pairs": int(len(d)),
        "current_family_counts": d["current_family"].value_counts().to_dict(),
        "prior_family_counts": d["prior_family"].value_counts().to_dict(),
        "same_exact_procedure_pairs": int(d["same_exact_procedure"].sum()),
        "same_exact_procedure_fraction": float(d["same_exact_procedure"].mean()),
        "same_broad_family_pairs": int(d["same_broad_family"].sum()),
        "same_broad_family_fraction": float(d["same_broad_family"].mean()),
        "rule_order": [family for family, _ in RULES],
        "target_or_map_columns_read": False,
        "interpretation": "Transparent coarse context mapping; not validated procedure coding.",
    }
    (args.output_dir / "outcome_blind_mapping_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
