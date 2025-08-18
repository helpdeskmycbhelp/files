import os
import re
import io
import csv
import glob
import traceback
from datetime import datetime, timedelta
from urllib.parse import urlencode

from dotenv import load_dotenv
load_dotenv()  # load .env before reading env vars

from flask import (
    Flask,
    render_template,
    render_template_string,
    request,
    jsonify,
    Response,
    make_response,
)
# if you use Flask 3.x and blueprint/templating specifics adjust imports accordingly
from werkzeug.exceptions import HTTPException
from pymongo import MongoClient, TEXT
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
from bson import ObjectId

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-only-not-secret")

# -----------------------------
# MongoDB Connection (local default)
# -----------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "property_db")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "properties")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
try:
    client.admin.command("ping")
except ServerSelectionTimeoutError as e:
    raise RuntimeError(f"Cannot connect to MongoDB at {MONGO_URI}. {e}")

db = client.get_database(DB_NAME)
collection = db[COLLECTION_NAME]

# -----------------------------
# Indexes (idempotent / safe to re-run)
# -----------------------------
try:
    # List view sorts
    collection.create_index(
        [
            ("city", 1),
            ("community", 1),
            ("sub_community", 1),
            ("property_type", 1),
            ("sub_type", 1),
            ("beds", 1),
            ("area_sqft", -1),
        ],
        name="loc_type_beds_area_desc",
    )
    collection.create_index(
        [
            ("city", 1),
            ("community", 1),
            ("sub_community", 1),
            ("property_type", 1),
            ("sub_type", 1),
            ("beds", 1),
            ("price", -1),
        ],
        name="loc_type_beds_price_desc",
    )
    # Municipality exacts
    collection.create_index(
        [("municipality_number", 1), ("municipality_sub_number", 1)],
        name="mun_munsub",
    )

    # Transactions primary sort keys (MULTIKEY)
    collection.create_index(
        [("owners.registration_date", -1), ("building_name", 1), ("unit_number", 1)],
        name="owners_regdate_desc__building__unit",
    )

    # Helpful singles
    for f in [
        "city","building_name","community","sub_community","property_type","sub_type",
        "municipality_number","municipality_sub_number","area_sqft","price","beds",
        "owners.owner_name","owners.registration_date",
    ]:
        collection.create_index(f)

    # Text index for free text on home()
    collection.create_index(
        [
            ("building_name", TEXT),
            ("community", TEXT),
            ("sub_community", TEXT),
            ("city", TEXT),
            ("owners.owner_name", TEXT),
            ("owners.contacts", TEXT),
        ],
        name="text_main",
        default_language="english",
    )
except Exception:
    pass

# -----------------------------
# Helpers
# -----------------------------
def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def _clean_text_list(vals):
    out = []
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in ("nan", "null", "none"):
            continue
        out.append(s)
    return sorted(set(out), key=lambda x: x.lower())

def _clean_beds_list(vals):
    cleaned = []
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in ("nan", "null", "none"):
            continue
        try:
            f = float(s)
            n = int(f) if float(f).is_integer() else f
            cleaned.append(n)
        except ValueError:
            pass
    return sorted(set(cleaned), key=lambda x: float(x))

def _regex_contains(text):
    if text is None:
        return None
    safe = re.escape(str(text))
    return {"$regex": safe, "$options": "i"}

def _fallback_image_for(prop_id_str: str) -> str:
    images = [
        "https://images.unsplash.com/photo-1560185127-6ed189bf02f4?q=80&w=1200&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1484154218962-a197022b5858?q=80&w=1200&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1580587771525-78b9dba3b914?q=80&w=1200&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1501183638710-841dd1904471?q=80&w=1200&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1556020685-ae41abfc9365?q=80&w=1200&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1460317442991-0ec209397118?q=80&w=1200&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1505691938895-1758d7feb511?q=80&w=1200&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1505691723518-36a5ac3be353?q=80&w=1200&auto=format&fit=crop",
    ]
    try:
        seed = int(str(prop_id_str)[-6:], 16)
    except Exception:
        seed = 0
    return images[seed % len(images)]

def _attach_hero_img(doc):
    if not isinstance(doc, dict):
        return doc
    img = (doc.get("image_url") or "").strip() if isinstance(doc.get("image_url"), str) else ""
    doc["hero_img"] = img if img else _fallback_image_for(str(doc.get("_id", "")))
    return doc

def _clean_opts(vals):
    out = []
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in ("none", "null", "nan"):
            continue
        out.append(s)
    return sorted(set(out), key=lambda x: x.lower())

def distinct_any(names, q=None):
    for n in names:
        vals = collection.distinct(n, q or {})
        cleaned = _clean_text_list(vals)
        if cleaned:
            return cleaned
    return []

def _parse_iso_date(s: str):
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except Exception:
        return None

def _pagination_window(total_pages: int, current_page: int, neighbors: int = 2):
    if total_pages <= 0:
        return [], None, None
    current_page = max(1, min(current_page, total_pages))
    start = max(1, current_page - neighbors)
    end = min(total_pages, current_page + neighbors)
    pages = list(range(start, end + 1))
    prev_page = current_page - 1 if current_page > 1 else None
    next_page = current_page + 1 if current_page < total_pages else None
    return pages, prev_page, next_page

# -----------------------------
# Health / Diagnostics / Favicon
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}

@app.get("/db-ping")
def db_ping():
    try:
        client.admin.command("ping")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

@app.get("/__diag")
def __diag():
    here = os.getcwd()
    tdir = os.path.abspath(getattr(app, "template_folder", "templates"))
    return {
        "cwd": here,
        "template_folder": tdir,
        "templates_found": sorted([os.path.basename(p) for p in glob.glob(os.path.join(tdir, "*.html"))]),
    }

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

# -----------------------------
# Small APIs
# -----------------------------
@app.get("/api/cities")
def api_cities():
    vals = _clean_opts(collection.distinct("city"))
    resp = make_response(jsonify({"cities": vals}))
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp

@app.get("/api/cascade_options")
def api_cascade_options():
    city = (request.args.get("city") or "").strip()
    bldg = (request.args.get("building_name") or "").strip()
    comm = (request.args.get("community") or "").strip()

    q = {}
    if city: q["city"] = city
    if bldg: q["building_name"] = bldg
    if comm: q["community"] = comm

    cur = collection.find(q, {"building_name": 1, "community": 1, "sub_community": 1})
    buildings, communities, subcomms = set(), set(), set()
    for doc in cur:
        if doc.get("building_name"):
            buildings.add(str(doc["building_name"]).strip())
        if doc.get("community"):
            communities.add(str(doc["community"]).strip())
        if doc.get("sub_community"):
            subcomms.add(str(doc["sub_community"]).strip())

    resp = jsonify(
        {"buildings": sorted(buildings), "communities": sorted(communities), "sub_communities": sorted(subcomms)}
    )
    h = make_response(resp)
    h.headers["Cache-Control"] = "public, max-age=900"
    return h

@app.get("/api/types")
def api_types():
    q = {}
    for key in ("city", "building_name", "community", "sub_community"):
        val = (request.args.get(key) or "").strip()
        if val:
            q[key] = val
    types_ = _clean_opts(collection.distinct("property_type", q))
    h = make_response(jsonify({"property_types": types_}))
    h.headers["Cache-Control"] = "public, max-age=900"
    return h

@app.get("/api/subtypes")
def api_subtypes():
    q = {}
    for key in ("city", "building_name", "community", "sub_community", "property_type"):
        val = (request.args.get(key) or "").strip()
        if val:
            q[key] = val
    sub_types = _clean_opts(collection.distinct("sub_type", q))
    h = make_response(jsonify({"sub_types": sub_types}))
    h.headers["Cache-Control"] = "public, max-age=900"
    return h

@app.get("/debug/distinct/<field>")
def debug_distinct(field):
    vals = list(collection.distinct(field))
    return jsonify({"count": len(vals), "sample": vals[:50]})

# -----------------------------
# Pages: HOME
# -----------------------------
@app.route("/")
def home():
    query = {}

    # Free-text search
    search = request.args.get("building")
    used_text_search = False
    if search:
        s = search.strip()
        if len(s) >= 3:
            query["$text"] = {"$search": s}
            used_text_search = True
        else:
            rx = _regex_contains(s)
            query["$or"] = [
                {"building_name": rx},
                {"community": rx},
                {"sub_community": rx},
                {"city": rx},
                {"owners.owner_name": rx},
                {"owners.contacts": rx},
            ]

    # Exact filters
    for key in ["property_type", "community", "city", "sub_community", "sub_type", "building_name"]:
        val = request.args.get(key)
        if val not in (None, ""):
            query[key] = val

    # Municipality numbers
    land_number_ui = request.args.get("land_number")
    mun_number_ui = request.args.get("municipality_number")
    final_mun_number = mun_number_ui or land_number_ui
    if final_mun_number not in (None, ""):
        query["municipality_number"] = final_mun_number

    mun_sub_ui = request.args.get("municipality_sub_number")
    if mun_sub_ui not in (None, ""):
        query["municipality_sub_number"] = mun_sub_ui

    # Beds exact
    beds_val = request.args.get("beds")
    if beds_val not in (None, ""):
        f = _to_float(beds_val)
        if f is not None:
            query["beds"] = f

    # Area range
    min_area = _to_float(request.args.get("min_area"))
    max_area = _to_float(request.args.get("max_area"))
    if min_area is not None or max_area is not None:
        rng = {}
        if min_area is not None: rng["$gte"] = min_area
        if max_area is not None: rng["$lte"] = max_area
        if rng: query["area_sqft"] = rng

    # Price range
    min_price = _to_float(request.args.get("min_price"))
    max_price = _to_float(request.args.get("max_price"))
    if min_price is not None or max_price is not None:
        rng = {}
        if min_price is not None: rng["$gte"] = min_price
        if max_price is not None: rng["$lte"] = max_price
        if rng: query["price"] = rng

    # Sorting
    sort_map = {"price": "price", "area_sqft": "area_sqft", "building_name": "building_name", "beds": "beds"}
    sort_req = request.args.get("sort_by") or "area_sqft"
    sort_field = sort_map.get(sort_req, "area_sqft")
    sort_order = request.args.get("order") or "desc"
    sort_dir = 1 if sort_order == "asc" else -1

    # Pagination (fixed per_page = 12 on home)
    try:
        page = int(request.args.get("page", 1) or 1)
        if page < 1:
            page = 1
    except ValueError:
        page = 1
    per_page = 12

    total_properties = collection.count_documents(query)
    total_pages = max((total_properties + per_page - 1) // per_page, 1)
    page = min(max(page, 1), total_pages)
    skip = (page - 1) * per_page

    # Projection for speed on list view
    PROPS_LIST_FIELDS = {
        "building_name": 1, "unit_number": 1, "beds": 1, "area_sqft": 1,
        "city": 1, "community": 1, "property_type": 1, "image_url": 1,
    }

    if used_text_search:
        cursor = (
            collection.find(query, {**PROPS_LIST_FIELDS, "score": {"$meta": "textScore"}})
            .sort([("score", {"$meta": "textScore"})])
            .skip(skip)
            .limit(per_page)
        )
    else:
        cursor = collection.find(query, PROPS_LIST_FIELDS).sort(sort_field, sort_dir).skip(skip).limit(per_page)

    properties = [_attach_hero_img(p) for p in cursor]

    # Dropdowns
    property_types = distinct_any(["property_type", "propertyType", "Property Type"])
    sub_types = distinct_any(["sub_type", "subType", "Sub Type"])
    communities = _clean_text_list(collection.distinct("community"))
    sub_communities = _clean_text_list(collection.distinct("sub_community"))
    cities = _clean_text_list(collection.distinct("city"))
    land_numbers = _clean_text_list(collection.distinct("municipality_number"))
    land_sub_numbers = _clean_text_list(collection.distinct("municipality_sub_number"))
    beds_list = _clean_beds_list(collection.distinct("beds"))

    total_communities = len(communities)
    total_cities = len(cities)
    total_count = total_properties

    # Query args for paginator links
    filters_qs = request.args.to_dict()
    filters_qs.pop("page", None)
    base_qs = urlencode(filters_qs)

    # Sliding window for page numbers
    page_numbers, prev_page, next_page = _pagination_window(total_pages, page, neighbors=2)
    last_page = total_pages

    ctx = dict(
        properties=properties,
        dropdowns={
            "property_types": property_types,
            "communities": communities,
            "sub_communities": sub_communities,
            "cities": cities,
            "sub_types": sub_types,
            "land_numbers": land_numbers,
            "land_sub_numbers": land_sub_numbers,
            "beds_list": beds_list,
        },
        property_types=property_types,
        communities=communities,
        sub_communities=sub_communities,
        cities=cities,
        sub_types=sub_types,
        beds_list=beds_list,
        land_numbers=land_numbers,
        land_sub_numbers=land_sub_numbers,
        total_pages=total_pages,
        current_page=page,
        last_page=last_page,          # expose last page
        total_count=total_count,
        total_communities=total_communities,
        total_cities=total_cities,
        used_text_search=used_text_search,
        base_qs=base_qs,
        page_numbers=page_numbers,
        prev_page=prev_page,
        next_page=next_page,
        per_page=per_page,
        query_args=filters_qs,
    )
    try:
        return render_template("index.html", **ctx)
    except Exception as e:
        fb = """
        <!doctype html><meta charset="utf-8">
        <h2>Index fallback (template failed)</h2>
        <p><b>Error:</b> {{err}}</p>
        <ul>{% for p in props[:10] %}<li>{{p.get('building_name')}} — {{p.get('unit_number')}} — {{p.get('city')}}</li>{% endfor %}</ul>
        <p><a href="/transactions">Go to Transactions</a></p>
        """
        return render_template_string(
            fb,
            err=f"{type(e).__name__}: {e}",
            props=properties,
        ), 200

# -----------------------------
# Page: Property detail
# -----------------------------
@app.route("/property/<property_id>")
def property_detail(property_id):
    prop = None
    try:
        prop = collection.find_one({"_id": ObjectId(property_id)})
    except Exception:
        prop = None
    if prop:
        _attach_hero_img(prop)
    return render_template("detail.html", prop=prop)

# -----------------------------
# Transactions aggregation builder
# -----------------------------
def _transactions_pipeline(match_stage, from_str, to_str):
    """
    UNWIND owners first so the multikey index on owners.registration_date can be used.
    Then:
      - Match date window on the unwound owner doc
      - Convert/choose price (owner > property)
      - Drop rows without a real price
      - Sort (reg_date desc, building asc, unit asc) -> index-backed
      - Group to a single row per (building, unit, date, price)
    """
    base = []
    if match_stage:
        base.append({"$match": match_stage})

    base += [
        {"$unwind": {"path": "$owners", "preserveNullAndEmptyArrays": False}},

        {"$match": {
            "owners.registration_date": {"$gte": from_str, "$lte": to_str}
        }},

        {"$project": {
            "building_name": 1,
            "unit_number": 1,
            "property_type": 1,
            "beds": 1,
            "area_sqft": 1,
            "city": 1,
            "community": 1,
            "sub_community": 1,

            "owner_name": "$owners.owner_name",
            "owner_role": {"$toLower": {"$ifNull": ["$owners.role", ""]}},
            "owner_price": {
                "$convert": {"input": "$owners.price", "to": "double", "onError": None, "onNull": None}
            },
            "prop_price": {
                "$convert": {"input": "$price", "to": "double", "onError": None, "onNull": None}
            },
            "reg_date": "$owners.registration_date",
        }},

        {"$addFields": {"price_eff": {"$ifNull": ["$owner_price", "$prop_price"]}}},
        {"$match": {"price_eff": {"$ne": None, "$gt": 0}}},

        {"$addFields": {
            "seller_name": {
                "$cond": [{"$regexMatch": {"input": "$owner_role", "regex": "seller"}}, "$owner_name", None]
            },
            "buyer_name": {
                "$cond": [{"$regexMatch": {"input": "$owner_role", "regex": "buyer"}}, "$owner_name", None]
            },
        }},

        {"$sort": {"reg_date": -1, "building_name": 1, "unit_number": 1}},

        {"$group": {
            "_id": {
                "building_name": "$building_name",
                "unit_number": "$unit_number",
                "date": "$reg_date",
                "price": "$price_eff",
            },
            "date": {"$first": "$reg_date"},
            "price": {"$first": "$price_eff"},
            "building_name": {"$first": "$building_name"},
            "unit_number": {"$first": "$unit_number"},
            "property_type": {"$first": "$property_type"},
            "beds": {"$first": "$beds"},
            "area_sqft": {"$first": "$area_sqft"},
            "city": {"$first": "$city"},
            "community": {"$first": "$community"},
            "sub_community": {"$first": "$sub_community"},
            "sellers": {"$addToSet": "$seller_name"},
            "buyers": {"$addToSet": "$buyer_name"},
        }},

        {"$addFields": {
            "sellers": {"$filter": {
                "input": "$sellers", "as": "s",
                "cond": {"$and": [{"$ne": ["$$s", None]}, {"$ne": ["$$s", ""]}]}
            }},
            "buyers": {"$filter": {
                "input": "$buyers", "as": "b",
                "cond": {"$and": [{"$ne": ["$$b", None]}, {"$ne": ["$$b", ""]}]}
            }},
        }},
        # keep the earlier sort order; no $sort after $group to avoid extra memory
    ]
    return base

# -----------------------------
# Page: Transactions
# -----------------------------
@app.route("/transactions")
def transactions():
    filters = request.args.to_dict(flat=True)

    # pagination (fixed 20 per page)
    try:
        current_page = max(1, int(request.args.get("page", 1) or 1))
    except ValueError:
        current_page = 1
    per_page = 20
    skip = (current_page - 1) * per_page

    # exact filters (case-insensitive exact)
    match_stage: dict = {}
    for key in ("building_name", "unit_number", "city", "community", "sub_community", "property_type", "sub_type"):
        v = (request.args.get(key) or "").strip()
        if v:
            match_stage[key] = {"$regex": f"^{re.escape(v)}$", "$options": "i"}

    # free text
    search = (request.args.get("q") or "").strip()
    if search:
        rx = {"$regex": re.escape(search), "$options": "i"}
        ors = [{"building_name": rx}, {"community": rx}, {"sub_community": rx}, {"city": rx}, {"owners.owner_name": rx}]
        match_stage = {"$and": [match_stage, {"$or": ors}]} if match_stage else {"$or": ors}

    # dates (strings 'YYYY-MM-DD')
    from_dt = _parse_iso_date(request.args.get("from") or "")
    to_dt   = _parse_iso_date(request.args.get("to") or "")
    if not from_dt and not to_dt:
        to_dt   = datetime.utcnow().date()
        from_dt = to_dt - timedelta(days=365)

    from_str = from_dt.strftime("%Y-%m-%d") if from_dt else "0000-01-01"
    to_str   = to_dt.strftime("%Y-%m-%d") if to_dt else "9999-12-31"

    # Build pipeline
    base = _transactions_pipeline(match_stage, from_str, to_str)

    # CSV export (unchanged)
    exporting = (request.args.get("export") or "").lower() == "csv"
    if exporting:
        cur = collection.aggregate(base, allowDiskUse=True, hint="owners_regdate_desc__building__unit")
        def generate():
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                ["date","building_name","unit_number","price","property_type","beds","area_sqft","city","community","sub_community","sellers","buyers"]
            )
            yield buf.getvalue(); buf.seek(0); buf.truncate(0)
            for r in cur:
                row = [
                    r.get("date",""), r.get("building_name",""), r.get("unit_number",""),
                    r.get("price",""), r.get("property_type",""), r.get("beds",""),
                    r.get("area_sqft",""), r.get("city",""), r.get("community",""),
                    r.get("sub_community",""),
                    "; ".join([s for s in (r.get("sellers") or []) if s]),
                    "; ".join([b for b in (r.get("buyers") or []) if b]),
                ]
                writer.writerow(row)
                yield buf.getvalue(); buf.seek(0); buf.truncate(0)
        return Response(generate(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=transactions.csv"})

    # ---------- COUNTED PAGINATION (HTML) ----------

    # 1) Count total grouped rows using the same base pipeline
    count_res = list(
        collection.aggregate(
            base + [{"$count": "total"}],
            allowDiskUse=True,
            hint="owners_regdate_desc__building__unit",
        )
    )
    total_count = int(count_res[0]["total"]) if count_res else 0

    # 2) Compute proper pagination numbers
    per_page = 20  # keep in sync with your setting above
    total_pages = max((total_count + per_page - 1) // per_page, 1)

    # Clamp current page and recompute skip after we know last page
    current_page = min(max(1, current_page), total_pages)
    skip = (current_page - 1) * per_page

    # 3) Fetch just this page worth of rows
    page_pipe = base + [{"$skip": skip}, {"$limit": per_page}]
    rows = list(
        collection.aggregate(
            page_pipe,
            allowDiskUse=True,
            hint="owners_regdate_desc__building__unit",
        )
    )

    # 4) Build windowed page numbers (+ prev/next) and last_page
    page_numbers, prev_page, next_page = _pagination_window(
        total_pages, current_page, neighbors=2
    )
    last_page = total_pages

    # 5) Build base_qs for links (exclude paging/export params)
    base_qs = urlencode({k: v for k, v in filters.items() if k not in ("page", "per_page", "export")})

    return render_template(
        "transactions.html",
        rows=rows,
        total=total_count,
        filters=filters,
        current_page=current_page,
        total_pages=total_pages,
        last_page=last_page,        # for "Last" button
        prev_page=prev_page,
        next_page=next_page,
        page_numbers=page_numbers,  # contains prev/next numbers around current
        per_page=per_page,
        base_qs=base_qs,
        clamped=False,
    )

# -----------------------------
# Error handling
# -----------------------------
@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):
        return e
    print("\n=== Unhandled Exception ===")
    print(f"Path: {request.path}")
    traceback.print_exc()
    is_dev = os.getenv("FLASK_ENV") == "development"
    if is_dev:
        return (f"500: {type(e).__name__}: {e}", 500)
    return ("Internal Server Error. Check server logs for details.", 500)

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
