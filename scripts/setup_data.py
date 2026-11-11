"""Create and seed the database, then build the document index.

    python -m scripts.setup_data
"""

from __future__ import annotations

import asyncio
import json

from sqlalchemy import create_engine

from retail_agent.config import get_settings
from retail_agent.db.seed import seed_database
from retail_agent.rag.ingest import build_index


def main() -> None:
    settings = get_settings()
    if settings.database_url.startswith("sqlite"):
        settings.docs_dir.parent.mkdir(parents=True, exist_ok=True)
    counts = seed_database(create_engine(settings.database_url))
    print("database:", json.dumps(counts))
    _, stats = asyncio.run(build_index(settings))
    print("index:", json.dumps(stats.__dict__))


if __name__ == "__main__":
    main()
