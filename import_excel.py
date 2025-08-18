#!/usr/bin/env python3
"""
Excel → MongoDB Importer (normalized buildings + transactions)

What's new vs your previous version:
- Normalizes/canonicalizes building names:
    * building_name_norm  = lowercased, trimmed, spaces collapsed
    * building_name_key   = letters+digits only (good for equality lookups)
    * building_name_canon = canonical display bucket ("23 marina" for all variants)
    * building_aliases    = list of raw variants we've seen for this unit
- Creates helpful indexes for fast filtering/sorting by canonical fields
- Keeps your owner merge logic and transactions[] with owner_snapshot
- Maintains last_price and last_transaction_date

Requirements:
    pip install pandas openpyxl pymongo python-dateutil
    (optional) pip install jellyfish   # for very conservative fuzzy correction
"""

from pathlib import Path
import re
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
from pymongo import MongoClient

# ===========================
# CONFIG — EDIT AS NEEDED
# ===========================
MONGO_URI  = "mongodb://localhost:27017"   # <- local default
DB_NAME    = "property_db"
COLL_NAME  = "properties"

EXCEL_PATH = "Dubai Marina.xlsx"  # full or relative path

# ===========================
# NORMALIZATION HELPERS
# ===========================
try:
    import jellyfish  # optional; used only if installed
except ImportError:
    jellyfish = None

# Hand-curated corrections you can grow over time
CANON_MAP = {
    "23 marnia": "23 marina",
    "23marina": "23 marina",
    "23  marina": "23 marina",
    "23 marina tower": "23 marina",
    "23marina": "23 marina",
      
        # example
    # add common typos you observe
}

def norm_text(s: str) -> str:
    """Lowercase, trim, and collapse internal whitespace."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def key_text(s: str) -> str:
    """Aggressive key for equality/grouping: letters+digits only."""
    return re.sub(r"[^a-z0-9]", "", norm_text(s))

def canon_building_name(raw: str) -> str:
    """
    Canonical display bucket for a building.
    1) normalize
    2) apply manual map
    3) (optional) fuzzy snap for very-close matches (strict threshold)
    """
    n = norm_text(raw)
    if not n:
        return ""

    if n in CANON_MAP:
        return CANON_MAP[n]

    # VERY conservative fuzzy correction (only if jellyfish is available)
    if jellyfish:
        # Seed with known canonical spellings you care about
        candidates = [
            "23 marina",
            # add more seeds here as needed
        ]
        best = None
        best_sim = 0.0
        for c in candidates:
            sim = jellyfish.jaro_winkler(n, c)
            if sim > best_sim:
                best, best_sim = c, sim
        if best and best_sim >= 0.97:  # strict to avoid wrong merges
            return best

    return n

# ===========================
# PARSERS
# ===========================
def parse_money(val) -> Optional[float]:
    """Return float or None. Handles 'AED 1,200,000', '1,25,000', etc."""
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in ("", "nan", "null", "none", "-"):
        return None
    s = re.sub(r"[^\d\.\-]", "", s)
    parts = s.split(".")
    if len(parts) > 2:
        s = parts[0] + "." + "".join(parts[1:])
    try:
        return float(s)
    except ValueError:
        return None

def parse_number(val) -> Optional[float | int]:
    """Generic numeric parser for area/beds/floor."""
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in ("", "nan", "null", "none", "-"):
        return None
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None

def parse_date(val) -> str:
    """Return ISO date 'YYYY-MM-DD' or ''."""
    if not val:
        return ""
    try:
        dt = pd.to_datetime(val, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return ""
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

def clean_phone(val: str) -> str:
    """Normalize phone: keep + and digits, 00→+, fix UAE 05… → +9715…"""
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ("", "nan", "null", "none", "-"):
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"[^\d+]", "", s)
    if not s:
        return ""
    if s.startswith("00"):
        s = "+" + s[2:]
    if s.startswith("05"):  # UAE local mobile
        s = "+971" + s[1:]
    return s

def split_contacts(raw: str) -> List[str]:
    """Split a 'Contact' cell that may contain multiple numbers."""
    if not raw:
        return []
    parts = re.split(r"[;,/|&\s]+", str(raw))
    out, seen = [], set()
    for p in parts:
        c = clean_phone(p)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def pick(row: pd.Series, candidates: List[str]) -> str:
    """
    Return the first non-empty value from row across candidate column names.
    - Case-insensitive exact match first, then partial contains match.
    """
    cols_lower = {str(c).lower(): c for c in row.index}

    # exact (case-insensitive)
    for cand in candidates:
        key = str(cand).lower()
        if key in cols_lower:
            v = row.get(cols_lower[key], "")
            if str(v).strip():
                return str(v).strip()

    # partial contains (case-insensitive)
    for cand in candidates:
        key = str(cand).lower()
        for col_low, original in cols_lower.items():
            if key in col_low:
                v = row.get(original, "")
                if str(v).strip():
                    return str(v).strip()

    return ""

def find_owner_indices(owners: List[dict], owner_name: str, role: str, reg_date: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (idx_same_date, idx_same_owner_any_date) where either can be None.
    idx_same_date: owner with same (name, role, registration_date)
    idx_same_owner_any_date: first owner with same (name, role) ignoring date
    """
    reg_date_norm = (reg_date or "")
    idx_same_date = next(
        (i for i, o in enumerate(owners)
         if o.get("owner_name") == owner_name
         and o.get("role") == role
         and (o.get("registration_date") or "") == reg_date_norm),
        None
    )
    idx_same_owner_any_date = next(
        (i for i, o in enumerate(owners)
         if o.get("owner_name") == owner_name
         and o.get("role") == role),
        None
    )
    return idx_same_date, idx_same_owner_any_date

def upsert_transaction(collection, doc_id, tx: dict):
    """
    Upsert a transaction (by date). If a transaction with the same date exists,
    update it; otherwise push a new one. Also refresh last_price and last_transaction_date when newer.
    """
    # 1) Try to update an existing transaction on same date
    res = collection.update_one(
        {"_id": doc_id, "transactions.date": tx["date"]},
        {"$set": {
            "transactions.$.price": tx.get("price"),
            "transactions.$.type": tx.get("type"),
            "transactions.$.beds": tx.get("beds"),
            "transactions.$.area_sqft": tx.get("area_sqft"),
            "transactions.$.floor": tx.get("floor"),
            "transactions.$.source": tx.get("source"),
            "transactions.$.notes": tx.get("notes"),
            "transactions.$.owner_snapshot": tx.get("owner_snapshot"),
        }}
    )

    if res.matched_count == 0:
        # 2) No transaction for that date → push new
        collection.update_one(
            {"_id": doc_id},
            {"$push": {"transactions": tx}}
        )

    # 3) Refresh convenience fields if this date is newer
    cur = collection.find_one({"_id": doc_id}, {"last_transaction_date": 1})
    old_date = (cur or {}).get("last_transaction_date") or ""
    if not old_date or tx["date"] > old_date:
        collection.update_one(
            {"_id": doc_id},
            {"$set": {
                "last_price": tx.get("price"),
                "last_transaction_date": tx["date"]
            }}
        )

# ===========================
# MAIN
# ===========================
def main():
    in_path = Path(EXCEL_PATH)
    if not in_path.exists():
        raise SystemExit(f"ERROR: Input file not found:\n  {in_path}")

    # ---- Load ALL sheets as text (preserves phone formatting) ----
    excel_book = pd.read_excel(
        in_path,
        sheet_name=None,
        dtype=str,              # keep everything as strings
        keep_default_na=False,  # don't convert "" to NaN
        na_filter=False
    )
    df = pd.concat(excel_book.values(), ignore_index=True)
    total_rows = len(df)

    # ---- DB & indexes ----
    client = MongoClient(MONGO_URI)
    db = client.get_database(DB_NAME)
    collection = db[COLL_NAME]

    # Useful indexes (idempotent)
    # Core identity
    collection.create_index([("building_name_canon", 1), ("unit_number", 1)])
    collection.create_index([("building_name_key", 1), ("unit_number", 1)])
    collection.create_index("building_name_norm")
    collection.create_index("building_name_key")
    collection.create_index("building_name_canon")

    # Owners / search helpers
    collection.create_index("owners.owner_name")
    collection.create_index("owners.contacts")
    collection.create_index("owners.registration_date")

    # Municipality
    collection.create_index("municipality_number")
    collection.create_index("municipality_sub_number")

    # Transactions + convenience
    collection.create_index("transactions.date")
    collection.create_index("last_transaction_date")

    # Cache existing to avoid repeated lookups
    existing_cache = {
        (doc["building_name_canon"], doc["unit_number"]): {
            "_id": doc["_id"],
            "owners": doc.get("owners", []),
            "display_building": doc.get("building_name", "")
        }
        for doc in collection.find(
            {},
            {"building_name": 1, "building_name_canon": 1, "unit_number": 1, "owners": 1}
        )
        if "building_name_canon" in doc and "unit_number" in doc
    }

    inserted = updated = 0
    owners_merged_contacts = 0
    owners_added_same_owner_new_date = 0
    owners_added_new_owner = 0
    tx_inserted = tx_updated = 0

    for _, row in df.iterrows():
        # === Robust field extraction (header variations supported) ===

        # Basic property:
        building_raw = pick(row, [
            "Building", "Building Name", "BuildingName", "BuildingNameEn",
            "Tower", "Tower Name", "Building (EN)"
        ])
        unit_number = pick(row, [
            "Unit No", "Unit no", "Unit Number", "UnitNumber", "Unit_No",
            "Unit-No", "Unit#", "Unit #", "Unit", "unitno", "unitno."
        ])
        area_sqft = parse_number(pick(row, ["Unit Size", "Size", "Area", "Area (sqft)", "Built-up Area"]))
        price_raw = pick(row, ["Price", "ProcedureValue", "Procedure Val", "ProcedureVal", "Value"])
        price = parse_money(price_raw)

        # Classification:
        property_type = (pick(row, ["Property Type", "PropertyType", "PropertyTypeEn"]) or None)
        sub_type      = (pick(row, ["Sub Type", "SubType", "SubTypeNameEn"]) or None)
        beds          = parse_number(pick(row, ["Beds", "Bed", "Bedrooms"]))

        # Location:
        city          = (pick(row, ["City"]) or None)
        community     = (pick(row, ["Community", "Project Lnd", "Project"]) or None)
        sub_community = (pick(row, ["Sub Community", "Sub-Community", "SubCommunity"]) or None)

        # Municipality (NOT land):
        municipality_number     = (pick(row, ["Mun No", "Municipality No", "Municipality Number"]) or None)
        municipality_sub_number = (pick(row, ["Mun Sub No", "Municipality Sub No", "Municipality Sub Number"]) or None)

        # Owner / transaction:
        owner_name = pick(row, ["Name", "NameEn", "Owner Name"])
        role       = pick(row, ["Role", "Owner Type", "ProcedurePartyTypeNameEn"])
        reg_date   = parse_date(pick(row, ["Regis", "Registration Date", "Reg Date"]))
        contacts   = split_contacts(pick(row, ["Contact", "Phone", "Mobile", "Whatsapp", "Tel"]))

        # Skip rows without core identifiers
        if not building_raw or not unit_number or not owner_name:
            continue

        # --- Normalize/canonicalize building name
        building_norm  = norm_text(building_raw)
        building_key   = key_text(building_raw)
        building_canon = canon_building_name(building_raw)

        owner_doc = {
            "owner_name": owner_name,
            "role": role,
            "contacts": contacts,
            "registration_date": reg_date
        }

        # NEW: build a transaction record per row (date + price required)
        tx_date = reg_date or parse_date(pick(row, ["Date", "Transaction Date"]))
        tx = None
        if tx_date and price is not None:
            tx = {
                "date": tx_date,
                "price": price,
                "type": property_type,
                "beds": beds,
                "area_sqft": area_sqft,
                "floor": parse_number(pick(row, ["Floor"])),
                "source": (pick(row, ["Source"]) or None),
                "notes": (pick(row, ["Notes", "Status"]) or None),
                "owner_snapshot": {
                    "owner_name": owner_name or None,
                    "role": role or None
                }
            }

        # Use (building_name_canon, unit_number) as the identity key
        key = (building_canon, unit_number)
        if key in existing_cache:
            # ----- UPDATE existing property -----
            doc_id = existing_cache[key]["_id"]
            owners = existing_cache[key].get("owners", [])

            # 1) refresh top-level + normalized fields when provided
            set_fields = {
                k: v for k, v in {
                    "building_name": building_raw,            # keep latest raw for display
                    "building_name_norm": building_norm,
                    "building_name_key": building_key,
                    "building_name_canon": building_canon,

                    "area_sqft": area_sqft,
                    "price": price,  # keep top-level price for compatibility
                    "price_raw": price_raw if price_raw else None,
                    "property_type": property_type,
                    "sub_type": sub_type,
                    "beds": beds,
                    "city": city,
                    "community": community,
                    "sub_community": sub_community,
                    "municipality_number": municipality_number,
                    "municipality_sub_number": municipality_sub_number,
                }.items() if v is not None and v != ""
            }
            if set_fields:
                collection.update_one({"_id": doc_id}, {"$set": set_fields})

            # maintain alias list for auditing
            collection.update_one({"_id": doc_id}, {"$addToSet": {"building_aliases": building_raw}})

            # 2) merge/append owners
            idx_same_date, idx_same_owner_any_date = find_owner_indices(
                owners, owner_name, role, reg_date
            )

            if idx_same_date is not None:
                # Same owner+role+date -> merge contacts only (no duplicate row)
                current = owners[idx_same_date]
                exist_contacts = set(current.get("contacts", []))
                new_nums = [c for c in contacts if c and c not in exist_contacts]

                if new_nums:
                    collection.update_one(
                        {
                            "_id": doc_id,
                            "owners.owner_name": owner_name,
                            "owners.role": role,
                            "owners.registration_date": reg_date or ""
                        },
                        {"$addToSet": {"owners.$.contacts": {"$each": new_nums}}}
                    )
                    # keep cache in sync
                    current.setdefault("contacts", []).extend(new_nums)
                    owners_merged_contacts += 1

            elif idx_same_owner_any_date is not None:
                # Same owner+role, different date -> push new dated entry
                collection.update_one({"_id": doc_id}, {"$push": {"owners": owner_doc}})
                owners.append(owner_doc)
                owners_added_same_owner_new_date += 1

            else:
                # Completely new owner for this property
                collection.update_one({"_id": doc_id}, {"$push": {"owners": owner_doc}})
                owners.append(owner_doc)
                owners_added_new_owner += 1

            # 3) upsert transaction for this row (if available)
            if tx:
                before = collection.find_one(
                    {"_id": doc_id, "transactions.date": tx["date"]},
                    {"_id": 1}
                )
                upsert_transaction(collection, doc_id, tx)
                if before:
                    tx_updated += 1
                else:
                    tx_inserted += 1

            updated += 1

        else:
            # ----- INSERT new property -----
            new_doc = {
                "building_name": building_raw,
                "building_name_norm": building_norm,
                "building_name_key": building_key,
                "building_name_canon": building_canon,
                "building_aliases": [building_raw],

                "unit_number": unit_number,
                "area_sqft": area_sqft,
                "price": price,
                "price_raw": price_raw if price_raw else None,
                "property_type": property_type,
                "sub_type": sub_type,
                "beds": beds,
                "city": city,
                "community": community,
                "sub_community": sub_community,
                "municipality_number": municipality_number,
                "municipality_sub_number": municipality_sub_number,
                "owners": [owner_doc],
            }
            # Seed transactions + convenience fields if we have tx
            if tx:
                new_doc["transactions"] = [tx]
                new_doc["last_price"] = tx["price"]
                new_doc["last_transaction_date"] = tx["date"]

            res = collection.insert_one(new_doc)
            existing_cache[key] = {"_id": res.inserted_id, "owners": [owner_doc], "display_building": building_raw}
            inserted += 1

            if tx:
                tx_inserted += 1

    # ---- Summary ----
    print("\n=== Import Summary ===")
    print(f"File: {in_path}")
    print(f"Total rows read: {total_rows}")
    print(f"Inserted properties: {inserted}")
    print(f"Updated properties:  {updated}")
    print(f"Owners merged (contacts-only): {owners_merged_contacts}")
    print(f"Owners added (same owner+role, NEW date): {owners_added_same_owner_new_date}")
    print(f"Owners added (new owner):               {owners_added_new_owner}")
    print(f"Transactions inserted: {tx_inserted}")
    print(f"Transactions updated:  {tx_updated}")

if __name__ == "__main__":
    main()
