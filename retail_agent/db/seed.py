"""Deterministic synthetic data for the Kestrel Retail warehouse.

Patterns are planted on purpose so that analytical questions have interesting answers and
match the policy documents in ``data/docs``:

* weekend uplift, category seasonality (beverages peak in summer, household in monsoon)
* promotion uplift that differs by campaign (some meet the playbook's 20% target, some do not)
* South-region stores grow ~3% a month
* a Dairy return spike in May 2026 (a supplier cold-chain failure, see the Q2 review)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
from sqlalchemy import Engine, func, insert, select

from retail_agent.db.models import metadata, products, promotions, returns, sales, stores

log = logging.getLogger(__name__)

SEED = 42
START, END = date(2026, 1, 1), date(2026, 9, 30)

STORES = [
    ("S01", "Kestrel Connaught Place", "Delhi", "North", "hypermarket", 42000, date(2019, 4, 1)),
    ("S02", "Kestrel Sector 17", "Chandigarh", "North", "supermarket", 18000, date(2020, 8, 15)),
    ("S03", "Kestrel Hazratganj", "Lucknow", "North", "express", 6000, date(2023, 2, 1)),
    ("S04", "Kestrel Andheri", "Mumbai", "West", "hypermarket", 45000, date(2018, 11, 1)),
    ("S05", "Kestrel Koregaon Park", "Pune", "West", "supermarket", 17000, date(2021, 6, 1)),
    ("S06", "Kestrel SG Highway", "Ahmedabad", "West", "supermarket", 16000, date(2022, 1, 10)),
    ("S07", "Kestrel Indiranagar", "Bengaluru", "South", "hypermarket", 40000, date(2019, 9, 1)),
    ("S08", "Kestrel T Nagar", "Chennai", "South", "supermarket", 19000, date(2021, 3, 1)),
    ("S09", "Kestrel Gachibowli", "Hyderabad", "South", "express", 7000, date(2024, 7, 1)),
    ("S10", "Kestrel Park Street", "Kolkata", "East", "supermarket", 17500, date(2020, 12, 1)),
]
FORMAT_FACTOR = {"hypermarket": 2.2, "supermarket": 1.0, "express": 0.45}

CATEGORIES = {
    # category: (brands, price range INR, cost ratio, base units/day/store)
    "Dairy": (["Gokul Fresh", "Amrit Dairy"], (30, 180), 0.78, 9.0),
    "Snacks": (["Crunchy Co", "Masala Bites"], (20, 120), 0.62, 7.0),
    "Beverages": (["Chill Spring", "Tea Garden"], (25, 220), 0.60, 6.0),
    "Personal Care": (["Pure Glow", "Herbal Roots"], (60, 450), 0.55, 3.0),
    "Household": (["Shine Max", "Home Guard"], (50, 400), 0.65, 3.5),
    "Fresh Produce": (["Farm Direct", "Green Valley"], (20, 150), 0.70, 10.0),
}
PRODUCT_NAMES = {
    "Dairy": ["Toned Milk 1L", "Curd 400g", "Paneer 200g", "Butter 100g",
              "Cheese Slices", "Lassi 200ml", "Ghee 500ml", "Flavoured Milk"],
    "Snacks": ["Potato Chips", "Namkeen Mix", "Cream Biscuits", "Roasted Peanuts",
               "Nachos", "Khakhra", "Chocolate Cookies", "Popcorn"],
    "Beverages": ["Mango Drink 1L", "Lemon Soda", "Green Tea 25", "Cold Coffee",
                  "Coconut Water", "Masala Chai 250g", "Orange Juice 1L", "Mineral Water 2L"],
    "Personal Care": ["Neem Face Wash", "Aloe Shampoo", "Herbal Soap 4pk", "Toothpaste 150g",
                      "Body Lotion", "Hair Oil", "Sunscreen SPF50", "Deodorant"],
    "Household": ["Dishwash Liquid", "Floor Cleaner", "Detergent 2kg", "Garbage Bags",
                  "Toilet Cleaner", "Mosquito Coil", "Air Freshener", "Kitchen Towels"],
    "Fresh Produce": ["Bananas 1kg", "Tomatoes 1kg", "Onions 1kg", "Apples 1kg",
                      "Spinach Bunch", "Potatoes 2kg", "Mangoes 1kg", "Cucumbers 500g"],
}
PROMOTIONS = [
    # id, name, category, discount, start, end, true demand uplift
    ("P01", "New Year Snack Fest", "Snacks", 0.20, date(2026, 1, 1), date(2026, 1, 15), 1.55),
    ("P02", "Holi Colours Care", "Personal Care", 0.15, date(2026, 3, 1), date(2026, 3, 14), 1.18),
    ("P03", "Summer Chill", "Beverages", 0.15, date(2026, 4, 15), date(2026, 5, 31), 1.50),
    ("P04", "Monsoon Home Saver", "Household", 0.10, date(2026, 7, 1), date(2026, 7, 21), 1.08),
    ("P05", "Fresh August", "Fresh Produce", 0.25, date(2026, 8, 1), date(2026, 8, 31), 1.60),
    ("P06", "Dairy Days", "Dairy", 0.12, date(2026, 9, 10), date(2026, 9, 20), 1.22),
]
SEASONALITY = {  # month -> category multiplier (default 1.0)
    "Beverages": {4: 1.25, 5: 1.35, 6: 1.30, 7: 1.05},
    "Household": {7: 1.20, 8: 1.15},
    "Fresh Produce": {5: 1.10, 6: 1.10},
    "Personal Care": {3: 1.08, 10: 1.10},
}
RETURN_RATE = {"Dairy": 0.012, "Snacks": 0.006, "Beverages": 0.008, "Personal Care": 0.010,
               "Household": 0.007, "Fresh Produce": 0.020}
RETURN_REASONS = {
    "Dairy": ["expired", "quality", "damaged"],
    "Fresh Produce": ["quality", "damaged", "expired"],
    "default": ["damaged", "wrong_item", "changed_mind", "quality"],
}


def _products(rng: np.random.Generator) -> list[dict]:
    rows, pid = [], 1
    for cat, (brands, (lo, hi), cost_ratio, _) in CATEGORIES.items():
        for i, name in enumerate(PRODUCT_NAMES[cat]):
            price = round(float(rng.uniform(lo, hi)), 0)
            rows.append({
                "product_id": pid,
                "sku": f"{cat[:3].upper()}-{pid:04d}",
                "product_name": f"{brands[i % 2]} {name}",
                "category": cat,
                "brand": brands[i % 2],
                "unit_price": price,
                "unit_cost": round(price * cost_ratio * float(rng.uniform(0.95, 1.05)), 2),
            })
            pid += 1
    return rows


def generate(seed: int = SEED) -> dict[str, list[dict]]:
    """Build all rows in memory (vectorised per store/product; ~130k sales lines)."""
    rng = np.random.default_rng(seed)
    prods = _products(rng)
    days = [START + timedelta(d) for d in range((END - START).days + 1)]
    n_days = len(days)
    months = np.array([d.month for d in days])
    weekend = np.array([1.25 if d.weekday() >= 5 else 1.0 for d in days])

    sales_rows: list[dict] = []
    return_rows: list[dict] = []
    sale_id = return_id = 1
    for store_id, *_rest, fmt, _area, _opened in STORES:
        region = _rest[2]
        store_mult = FORMAT_FACTOR[fmt] * float(rng.uniform(0.9, 1.1))
        growth = (1.0 + 0.03 * (months - 1)) if region == "South" else np.ones(n_days)
        for p in prods:
            cat = p["category"]
            base = CATEGORIES[cat][3] * float(rng.uniform(0.6, 1.4))
            season = np.array([SEASONALITY.get(cat, {}).get(m, 1.0) for m in months])
            promo_ids = np.full(n_days, None, dtype=object)
            disc = np.zeros(n_days)
            uplift = np.ones(n_days)
            for pr_id, _n, pr_cat, pct, s, e, up in PROMOTIONS:
                if pr_cat != cat:
                    continue
                mask = np.array([s <= d <= e for d in days])
                promo_ids[mask], disc[mask], uplift[mask] = pr_id, pct, up
            lam = base * store_mult * weekend * season * growth * uplift
            units = rng.poisson(lam)
            price = p["unit_price"]
            rr = np.full(n_days, RETURN_RATE[cat])
            if cat == "Dairy":
                rr[months == 5] = 0.062  # cold-chain failure at a supplier in May
            returned = rng.binomial(units, rr)
            reasons = RETURN_REASONS.get(cat, RETURN_REASONS["default"])
            for i in np.nonzero(units)[0]:
                u = int(units[i])
                gross = u * price
                sales_rows.append({
                    "sale_id": sale_id,
                    "sale_date": days[i],
                    "store_id": store_id,
                    "product_id": p["product_id"],
                    "promo_id": promo_ids[i],
                    "units": u,
                    "revenue": round(gross * (1 - disc[i]), 2),
                    "discount": round(gross * disc[i], 2),
                })
                if returned[i]:
                    return_rows.append({
                        "return_id": return_id,
                        "sale_id": sale_id,
                        "return_date": days[i] + timedelta(int(rng.integers(0, 8))),
                        "units": int(returned[i]),
                        "reason": reasons[int(rng.integers(0, len(reasons)))],
                    })
                    return_id += 1
                sale_id += 1

    return {
        "stores": [dict(zip(["store_id", "store_name", "city", "region", "format", "area_sqft",
                             "opened_on"], s, strict=True)) for s in STORES],
        "products": prods,
        "promotions": [dict(zip(["promo_id", "promo_name", "category", "discount_pct",
                                 "start_date", "end_date"], p[:6], strict=True))
                       for p in PROMOTIONS],
        "sales": sales_rows,
        "returns": return_rows,
    }


def is_seeded(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            return (conn.execute(select(func.count()).select_from(sales)).scalar() or 0) > 0
    except Exception:  # table missing
        return False


def seed_database(engine: Engine, seed: int = SEED, batch: int = 5000) -> dict[str, int]:
    """Create tables and load the synthetic data. Idempotent: drops and recreates the tables."""
    data = generate(seed)
    metadata.drop_all(engine)
    metadata.create_all(engine)
    order = [("stores", stores), ("products", products), ("promotions", promotions),
             ("sales", sales), ("returns", returns)]
    with engine.begin() as conn:
        for name, table in order:
            rows = data[name]
            for i in range(0, len(rows), batch):
                conn.execute(insert(table), rows[i : i + batch])
    counts = {name: len(data[name]) for name, _ in order}
    log.info("database seeded", extra={"fields": counts})
    return counts
