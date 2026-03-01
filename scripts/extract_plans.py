"""
Extract execution plans from PostgreSQL.

Runs parameterized queries with planner hints to force alternative plans,
then saves plans (including actual execution times) as JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import load_config

logger = logging.getLogger("neural_optimizer")


def connect_db(
    host: str = "localhost",
    port: int = 5432,
    dbname: str = "tpch",
    user: str | None = None,
    password: str | None = None,
) -> Any:
    """Create database connection."""
    import psycopg2

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user or "postgres",
        password=password,
    )
    return conn


def get_plan(conn: Any, sql: str, analyze: bool = True) -> dict | None:
    """
    Run EXPLAIN and return plan as JSON.
    With analyze=True, executes query and records actual rows/times.
    """
    cursor = conn.cursor()
    try:
        explain_sql = f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}"
        cursor.execute(explain_sql)
        row = cursor.fetchone()
        if row and row[0]:
            # EXPLAIN returns list of one plan dict
            plans = row[0]
            if isinstance(plans, list) and plans:
                return plans[0]["Plan"]
            return plans if isinstance(plans, dict) else None
    except Exception as e:
        logger.warning("Query failed: %s", e)
    finally:
        cursor.close()
    return None


def apply_planner_hints(cursor: Any, hints: dict[str, bool]) -> None:
    """Set planner GUCs to force alternative plans."""
    mapping = {
        "enable_hashjoin": "enable_hashjoin",
        "enable_mergejoin": "enable_mergejoin",
        "enable_nestloop": "enable_nestloop",
        "enable_indexscan": "enable_indexscan",
        "enable_seqscan": "enable_seqscan",
    }
    for key, val in hints.items():
        if key in mapping:
            cursor.execute(f"SET {mapping[key]} = {str(val).lower()}")


def extract_plan_recursive(plan: dict) -> dict:
    """Recursively extract plan subtree (flatten for storage)."""
    node = {
        "Node Type": plan.get("Node Type", "Unknown"),
        "Relation Name": plan.get("Relation Name"),
        "Alias": plan.get("Alias"),
        "Startup Cost": plan.get("Startup Cost"),
        "Total Cost": plan.get("Total Cost"),
        "Plan Rows": plan.get("Plan Rows"),
        "Plan Width": plan.get("Plan Width"),
        "Actual Startup Time": plan.get("Actual Startup Time"),
        "Actual Total Time": plan.get("Actual Total Time"),
        "Actual Rows": plan.get("Actual Rows"),
    }
    children = plan.get("Plans", [])
    if children:
        node["Plans"] = [extract_plan_recursive(c) for c in children]
    return node


def extract_plans(
    conn_config: dict,
    queries: list[dict],
    planner_hints: list[dict],
    output_dir: Path,
) -> list[Path]:
    """
    Extract plans for all query × hint combinations.
    Returns list of saved plan file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    conn = connect_db(**conn_config)

    for q_idx, qdef in enumerate(queries):
        sql = qdef.get("sql", "")
        if not sql:
            continue

        for h_idx, hints in enumerate(planner_hints):
            cursor = conn.cursor()
            try:
                apply_planner_hints(cursor, hints)
            finally:
                cursor.close()

            plan = get_plan(conn, sql)
            if plan is None:
                continue

            record = {
                "query_id": q_idx,
                "hint_id": h_idx,
                "sql": sql,
                "hints": hints,
                "plan": extract_plan_recursive(plan),
                "planning_time": plan.get("Planning Time"),
                "execution_time": plan.get("Execution Time"),
            }
            fname = output_dir / f"plan_q{q_idx}_h{h_idx}.json"
            with open(fname, "w") as f:
                json.dump(record, f, indent=2)
            saved.append(fname)
            logger.info("Saved %s (runtime=%.2f ms)", fname.name, record.get("execution_time") or 0)

    conn.close()
    return saved


# --- TPC-H style query templates (parameterized for diversity) ---
TPCH_QUERIES = [
    {
        "name": "Q1",
        "sql": """
        SELECT l_returnflag, l_linestatus, sum(l_quantity) as sum_qty
        FROM lineitem
        WHERE l_shipdate <= date '1998-12-01'
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
        """,
    },
    {
        "name": "Q3",
        "sql": """
        SELECT l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < date '1995-03-15'
          AND l_shipdate > date '1995-03-15'
        GROUP BY l_orderkey
        ORDER BY revenue DESC
        LIMIT 10
        """,
    },
    {
        "name": "Q5",
        "sql": """
        SELECT n_name, sum(l_extendedprice * (1 - l_discount)) as revenue
        FROM customer, orders, lineitem, supplier, nation, region
        WHERE c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND l_suppkey = s_suppkey
          AND c_nationkey = s_nationkey
          AND s_nationkey = n_nationkey
          AND n_regionkey = r_regionkey
          AND r_name = 'ASIA'
          AND o_orderdate >= date '1994-01-01'
          AND o_orderdate < date '1995-01-01'
        GROUP BY n_name
        ORDER BY revenue DESC
        """,
    },
    {
        "name": "Q6",
        "sql": """
        SELECT sum(l_extendedprice * l_discount) as revenue
        FROM lineitem
        WHERE l_shipdate >= date '1994-01-01'
          AND l_shipdate < date '1995-01-01'
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24
        """,
    },
]


def main() -> None:
    """CLI entry point."""
    import argparse

    from src.logging_config import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description="Extract PostgreSQL execution plans")
    parser.add_argument("--config", default="config/default.yaml", help="Config path")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="tpch")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--output", help="Override output dir")
    parser.add_argument("--queries", nargs="+", help="Query indices to run (default: all)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    output_dir = Path(args.output or data_cfg.get("output_dir", "data/plans"))
    hints = data_cfg.get("planner_hints", [{"enable_hashjoin": False, "enable_mergejoin": False}])

    conn_config = {"host": args.host, "port": args.port, "dbname": args.dbname, "user": args.user}
    queries = TPCH_QUERIES
    if args.queries:
        indices = [int(x) for x in args.queries]
        queries = [TPCH_QUERIES[i] for i in indices if 0 <= i < len(TPCH_QUERIES)]

    paths = extract_plans(conn_config, queries, hints, output_dir)
    logger.info("Extracted %d plans to %s", len(paths), output_dir)
