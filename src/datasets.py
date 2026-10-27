"""Unified dataset loader: ``load_dataset("adult")`` or ``load_dataset("acs_income")``.

Both return ``(df, config)`` where ``df`` is a cleaned DataFrame with readable
column names and categorical labels, and ``config`` describes the columns so the
modelling / plotting code never hard-codes a dataset's schema.

ACS code labels were taken from the official 2018 ACS PUMS Data Dictionary
(https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "01_raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "02_interim"
CONFIG_DIR = PROJECT_ROOT / "config"

# ---------------------------------------------------------------- Adult (UCI)
ADULT_RENAME = {
    "age": "Age", "workclass": "Class of Worker", "fnlwgt": "Sampling Weight",
    "education": "Education", "education-num": "Education Years",
    "marital-status": "Marital Status", "occupation": "Occupation",
    "relationship": "Relationship", "race": "Race", "sex": "Sex",
    "capital-gain": "Capital Gain", "capital-loss": "Capital Loss",
    "hours-per-week": "Weekly Work Hours", "native-country": "Birthplace",
    "income": "Income",
}

# ---------------------------------------------------------- ACS PUMS codebook
SEX_MAP = {1: "Male", 2: "Female"}
RAC1P_MAP = {
    1: "White", 2: "Black", 3: "American Indian", 4: "Alaska Native",
    5: "AIAN tribe / not specified", 6: "Asian", 7: "NHPI",
    8: "Some Other Race", 9: "Two or More Races",
}
MAR_MAP = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated",
           5: "Never married"}
COW_MAP = {
    1: "Private for-profit", 2: "Private not-for-profit", 3: "Local government",
    4: "State government", 5: "Federal government",
    6: "Self-employed (not inc.)", 7: "Self-employed (inc.)",
    8: "Without pay (family business)", 9: "Unemployed (5+ yrs)",
}
RELP_MAP = {
    0: "Reference person", 1: "Husband/wife", 2: "Biological child",
    3: "Adopted child", 4: "Stepchild", 5: "Sibling", 6: "Parent",
    7: "Grandchild", 8: "Parent-in-law", 9: "Child-in-law",
    10: "Other relative", 11: "Roomer/boarder", 12: "Housemate/roommate",
    13: "Unmarried partner", 14: "Foster child", 15: "Other nonrelative",
    16: "Institutional group quarters", 17: "Non-institutional group quarters",
}
OCCP_GROUP_NAMES = {
    "MGR": "Management", "BUS": "Business Ops", "FIN": "Finance",
    "CMM": "Computer & Math", "ENG": "Engineering", "SCI": "Science",
    "CMS": "Community & Social", "LGL": "Legal", "EDU": "Education",
    "ENT": "Arts, Media & Sports", "MED": "Healthcare Practitioner",
    "HLS": "Healthcare Support", "PRT": "Protective Service",
    "EAT": "Food Service", "CLN": "Cleaning & Maintenance",
    "PRS": "Personal Care", "SAL": "Sales", "OFF": "Office & Admin",
    "FFF": "Farming, Fishing & Forestry", "CON": "Construction",
    "EXT": "Extraction", "RPR": "Installation & Repair", "PRD": "Production",
    "TRN": "Transportation", "MIL": "Military", "UNEMP": "Unemployed (5+ yrs)",
}


def _schl_band(code: float) -> str:
    """Group SCHL (1-24) into readable education bands."""
    c = int(code)
    if c <= 15:
        return "Less than HS"
    if c <= 17:
        return "HS diploma / GED"
    if c <= 19:
        return "Some college"
    return {20: "Associate's", 21: "Bachelor's", 22: "Master's",
            23: "Professional", 24: "Doctorate"}[c]


SCHL_ORDER = ["Less than HS", "HS diploma / GED", "Some college", "Associate's",
              "Bachelor's", "Master's", "Professional", "Doctorate"]


def _pobp_region(code: float) -> str:
    """Coarse POBP region using the PUMS code blocks."""
    c = int(code)
    if c <= 56:
        return "US (state)"
    if c <= 78:
        return "US territory"
    if c <= 169:
        return "Europe"
    if c <= 254:
        return "Asia"
    if c == 303:
        return "Mexico"
    if c <= 399:
        return "Other Americas"
    if c <= 469:
        return "Africa"
    return "Oceania / other"


# ------------------------------------------------------------------ loaders
def _load_adult():
    path = RAW_DIR / "adult.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        from ucimlrepo import fetch_ucirepo
        adult = fetch_ucirepo(id=2)
        df = pd.concat([adult.data.features, adult.data.targets], axis=1)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    n_raw = len(df)
    # Same cleaning as notebooks/01_data_cleaning.ipynb
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df = df.replace("?", np.nan).dropna().drop_duplicates().copy()
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    df = df.drop(columns=["fnlwgt"]).rename(columns=ADULT_RENAME)
    df = df.reset_index(drop=True)
    edu_order = (df.groupby("Education")["Education Years"].median()
                 .sort_values().index.tolist())
    cfg = dict(
        name="adult", title="UCI Adult (1994 CPS)",
        numeric_cols=["Age", "Education Years", "Capital Gain", "Capital Loss",
                      "Weekly Work Hours"],
        categorical_cols=["Class of Worker", "Education", "Marital Status",
                          "Occupation", "Relationship", "Race", "Sex", "Birthplace"],
        target="Income", positive_label=">50K", negative_label="<=50K",
        target_desc="income > $50K",
        sensitive_cols=["Sex", "Race"],
        category_orders={"Education": edu_order},
        n_raw=n_raw,
    )
    return df, cfg


def _load_acs(state="CA", year="2018", horizon="1-Year", threshold=50_000):
    from folktables import ACSDataSource, ACSIncome
    root = RAW_DIR / "acs"
    src = ACSDataSource(survey_year=str(year), horizon=horizon, survey="person",
                        root_dir=str(root))
    states = [state] if isinstance(state, str) else list(state)
    raw = src.get_data(states=states, download=True)
    X, y, _ = ACSIncome.df_to_pandas(raw)  # ACSIncome filter: age>16, WKHP>=1, PINCP>100, PWGTP>=1
    n_raw = len(raw)
    df = X.copy()
    df["SEX"] = df["SEX"].astype(int).map(SEX_MAP)
    df["RAC1P"] = df["RAC1P"].astype(int).map(RAC1P_MAP)
    df["MAR"] = df["MAR"].astype(int).map(MAR_MAP)
    df["COW"] = df["COW"].astype(int).map(COW_MAP)
    df["RELP"] = df["RELP"].astype(int).map(RELP_MAP)
    df["SCHL"] = df["SCHL"].astype(int).map(_schl_band)
    occ = pd.read_csv(CONFIG_DIR / "acs_occp_groups.csv", dtype={"code": int})
    occ_map = dict(zip(occ["code"], occ["group"].map(OCCP_GROUP_NAMES)))
    df["OCCP"] = df["OCCP"].astype(int).map(occ_map).fillna("Other / unknown")
    df["POBP"] = df["POBP"].astype(int).map(_pobp_region)
    label = (y.iloc[:, 0].astype(bool) if threshold == 50_000
             else raw.loc[X.index, "PINCP"] > threshold)
    df["Income"] = np.where(label.values, ">50K", "<=50K")
    df = df.rename(columns={
        "AGEP": "Age", "COW": "Class of Worker", "SCHL": "Education",
        "MAR": "Marital Status", "OCCP": "Occupation", "POBP": "Birthplace",
        "RELP": "Relationship", "WKHP": "Weekly Work Hours", "SEX": "Sex",
        "RAC1P": "Race"}).reset_index(drop=True)
    df["Age"] = df["Age"].astype(int)
    df["Weekly Work Hours"] = df["Weekly Work Hours"].astype(int)
    cfg = dict(
        name="acs_income", title=f"ACSIncome ({states and ','.join(states)} {year}, {horizon})",
        numeric_cols=["Age", "Weekly Work Hours"],
        categorical_cols=["Class of Worker", "Education", "Marital Status",
                          "Occupation", "Relationship", "Race", "Sex", "Birthplace"],
        target="Income", positive_label=">50K", negative_label="<=50K",
        target_desc=f"personal income (PINCP) > ${threshold:,}",
        sensitive_cols=["Sex", "Race"],
        category_orders={"Education": SCHL_ORDER},
        n_raw=n_raw,
    )
    return df, cfg


def load_dataset(name: str = "adult", state="CA", year="2018", **kw):
    """Return ``(df, config)`` for ``name`` in {"adult", "acs_income"}.

    ``state`` / ``year`` only apply to ``acs_income`` (any folktables state code
    or list of codes; survey years 2014-2018 are supported by folktables).
    """
    if name == "adult":
        return _load_adult()
    if name == "acs_income":
        return _load_acs(state=state, year=year, **kw)
    raise ValueError(f"unknown dataset {name!r}; use 'adult' or 'acs_income'")
