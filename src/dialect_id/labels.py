from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DialectLabel:
    id: int
    code: str
    country: str
    region: str


DIALECTS: tuple[DialectLabel, ...] = (
    DialectLabel(0, "AE", "United Arab Emirates", "gulf"),
    DialectLabel(1, "BH", "Bahrain", "gulf"),
    DialectLabel(2, "DJ", "Djibouti", "east_africa"),
    DialectLabel(3, "DZ", "Algeria", "maghreb"),
    DialectLabel(4, "EG", "Egypt", "nile"),
    DialectLabel(5, "IQ", "Iraq", "iraqi"),
    DialectLabel(6, "JO", "Jordan", "levant"),
    DialectLabel(7, "KM", "Comoros", "east_africa"),
    DialectLabel(8, "KW", "Kuwait", "gulf"),
    DialectLabel(9, "LB", "Lebanon", "levant"),
    DialectLabel(10, "LY", "Libya", "maghreb"),
    DialectLabel(11, "MA", "Morocco", "maghreb"),
    DialectLabel(12, "MR", "Mauritania", "maghreb"),
    DialectLabel(13, "OM", "Oman", "gulf"),
    DialectLabel(14, "PS", "Palestine", "levant"),
    DialectLabel(15, "QA", "Qatar", "gulf"),
    DialectLabel(16, "SA", "Saudi Arabia", "gulf"),
    DialectLabel(17, "SD", "Sudan", "nile"),
    DialectLabel(18, "SO", "Somalia", "east_africa"),
    DialectLabel(19, "SY", "Syria", "levant"),
    DialectLabel(20, "TD", "Chad", "sahel"),
    DialectLabel(21, "TN", "Tunisia", "maghreb"),
    DialectLabel(22, "YE", "Yemen", "gulf"),
)

ID_TO_CODE = {label.id: label.code for label in DIALECTS}
CODE_TO_ID = {label.code: label.id for label in DIALECTS}
ID_TO_COUNTRY = {label.id: label.country for label in DIALECTS}
CODE_TO_COUNTRY = {label.code: label.country for label in DIALECTS}
CODE_TO_REGION = {label.code: label.region for label in DIALECTS}
ID_TO_REGION = {label.id: label.region for label in DIALECTS}

REGIONS = tuple(sorted({label.region for label in DIALECTS}))
REGION_TO_ID = {region: idx for idx, region in enumerate(REGIONS)}
ID_TO_REGION_NAME = {idx: region for region, idx in REGION_TO_ID.items()}
DIALECT_ID_TO_REGION_ID = {
    label.id: REGION_TO_ID[label.region] for label in DIALECTS
}

SPECIALIST_GROUPS: dict[str, tuple[str, ...]] = {
    "maghreb": ("MA", "DZ", "TN", "LY", "MR"),
    "levant": ("LB", "SY", "PS", "JO"),
    "gulf": ("AE", "BH", "KW", "QA", "OM", "SA", "YE"),
    "arabian": ("SA", "YE", "OM"),
    "nile": ("EG", "SD"),
    "east_africa": ("DJ", "KM", "SO"),
}
SPECIALIST_DIALECT_IDS = {
    name: tuple(CODE_TO_ID[code] for code in codes)
    for name, codes in SPECIALIST_GROUPS.items()
}
SPECIALIST_ID_TO_LOCAL = {
    name: {dialect_id: local_idx for local_idx, dialect_id in enumerate(dialect_ids)}
    for name, dialect_ids in SPECIALIST_DIALECT_IDS.items()
}


def num_dialects() -> int:
    return len(DIALECTS)


def num_regions() -> int:
    return len(REGIONS)


def label_payload() -> dict[str, object]:
    return {
        "dialects": [label.__dict__ for label in DIALECTS],
        "regions": list(REGIONS),
        "dialect_id_to_region_id": DIALECT_ID_TO_REGION_ID,
        "specialist_groups": {
            name: list(codes) for name, codes in SPECIALIST_GROUPS.items()
        },
    }
