"""Relational schema for the (synthetic) Kestrel Retail sales warehouse.

A small star schema: one fact table (``sales``) with dimensions for stores, products and
promotions, plus a ``returns`` table linked to individual sales. Column comments double as
the schema description given to the LLM for text-to-SQL.
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    Date,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
)

metadata = MetaData()

stores = Table(
    "stores",
    metadata,
    Column("store_id", String(4), primary_key=True, comment="Store code, e.g. S01"),
    Column("store_name", String(80), nullable=False),
    Column("city", String(40), nullable=False),
    Column("region", String(10), nullable=False, comment="North, South, East or West"),
    Column("format", String(15), nullable=False, comment="hypermarket, supermarket or express"),
    Column("area_sqft", Integer, nullable=False, comment="Selling area in square feet"),
    Column("opened_on", Date, nullable=False),
    comment="One row per physical store.",
)

products = Table(
    "products",
    metadata,
    Column("product_id", Integer, primary_key=True),
    Column("sku", String(12), nullable=False, unique=True),
    Column("product_name", String(80), nullable=False),
    Column(
        "category",
        String(20),
        nullable=False,
        comment="Dairy, Snacks, Beverages, Personal Care, Household or Fresh Produce",
    ),
    Column("brand", String(40), nullable=False),
    Column("unit_price", Numeric(10, 2), nullable=False, comment="List price in INR"),
    Column("unit_cost", Numeric(10, 2), nullable=False, comment="Purchase cost in INR"),
    comment="Product catalogue.",
)

promotions = Table(
    "promotions",
    metadata,
    Column("promo_id", String(4), primary_key=True),
    Column("promo_name", String(60), nullable=False),
    Column("category", String(20), nullable=False, comment="Category the discount applies to"),
    Column("discount_pct", Numeric(4, 2), nullable=False, comment="0.15 means 15% off"),
    Column("start_date", Date, nullable=False),
    Column("end_date", Date, nullable=False, comment="Inclusive"),
    comment="Category-wide promotional campaigns in 2026.",
)

sales = Table(
    "sales",
    metadata,
    Column("sale_id", Integer, primary_key=True),
    Column("sale_date", Date, nullable=False),
    Column("store_id", String(4), ForeignKey("stores.store_id"), nullable=False),
    Column("product_id", Integer, ForeignKey("products.product_id"), nullable=False),
    Column(
        "promo_id",
        String(4),
        ForeignKey("promotions.promo_id"),
        nullable=True,
        comment="Set when the line was sold under a promotion",
    ),
    Column("units", Integer, nullable=False),
    Column("revenue", Numeric(12, 2), nullable=False, comment="Net revenue in INR after discount"),
    Column("discount", Numeric(12, 2), nullable=False, comment="Discount given in INR"),
    comment="Daily sales fact: one row per store, product and day (1 Jan - 30 Sep 2026).",
)

returns = Table(
    "returns",
    metadata,
    Column("return_id", Integer, primary_key=True),
    Column("sale_id", Integer, ForeignKey("sales.sale_id"), nullable=False),
    Column("return_date", Date, nullable=False),
    Column("units", Integer, nullable=False, comment="Units returned"),
    Column(
        "reason",
        String(30),
        nullable=False,
        comment="damaged, expired, quality, wrong_item or changed_mind",
    ),
    comment="Customer returns, each linked to the original sales line.",
)

Index("ix_sales_date", sales.c.sale_date)
Index("ix_sales_store_date", sales.c.store_id, sales.c.sale_date)
Index("ix_sales_product", sales.c.product_id)
Index("ix_returns_sale", returns.c.sale_id)

ALLOWED_TABLES = frozenset(metadata.tables)
