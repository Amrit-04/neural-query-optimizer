"""
Extract Real PostgreSQL Execution Plans and Timings on TPC-H SF 0.05.

Extracts plain EXPLAIN plans for feature encoding, and uses ANALYZE only for the
execution time target. Follows strict benchmarking protocol:
- max_parallel_workers_per_gather = 0 (sequential execution)
- 4000ms statement_timeout (timed-out plans capped at 4000ms, marked timed_out=True)
- Discard first cold run, take median of 3 subsequent runs
- Deduplicate identical plan structures per query instance
- Detect and report templates with < 2 distinct plans
- Output saved to data/plans_real/ and results/real_data_summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from src.logging_config import setup_logging

logger = logging.getLogger("neural_optimizer")

# 12 TPC-H Query Templates with parameter binding
TEMPLATES: dict[int, dict[str, Any]] = {
    2: {
        "name": "Minimum Cost Supplier",
        "sql": """
            SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
            FROM part, supplier, partsupp, nation, region
            WHERE p_partkey = ps_partkey
              AND s_suppkey = ps_suppkey
              AND p_size = {size}
              AND p_type LIKE '%{type}'
              AND s_nationkey = n_nationkey
              AND n_regionkey = r_regionkey
              AND r_name = '{region}'
            ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
            LIMIT 10;
        """,
        "instances": [
            {"size": 15, "type": "BRASS", "region": "EUROPE"},
            {"size": 25, "type": "COPPER", "region": "ASIA"},
        ],
    },
    3: {
        "name": "Shipping Priority",
        "sql": """
            SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority
            FROM customer, orders, lineitem
            WHERE c_mktsegment = '{mktsegment}'
              AND c_custkey = o_custkey
              AND l_orderkey = o_orderkey
              AND o_orderdate < DATE '{date}'
              AND l_shipdate > DATE '{date}'
            GROUP BY l_orderkey, o_orderdate, o_shippriority
            ORDER BY revenue DESC, o_orderdate
            LIMIT 10;
        """,
        "instances": [
            {"mktsegment": "BUILDING", "date": "1995-03-15"},
            {"mktsegment": "AUTOMOBILE", "date": "1995-03-20"},
        ],
    },
    4: {
        "name": "Order Priority Checking",
        "sql": """
            SELECT o_orderpriority, COUNT(*) AS order_count
            FROM orders
            WHERE o_orderdate >= DATE '{date}'
              AND o_orderdate < DATE '{date}' + INTERVAL '3 month'
              AND EXISTS (
                  SELECT * FROM lineitem
                  WHERE l_orderkey = o_orderkey
                    AND l_commitdate < l_receiptdate
              )
            GROUP BY o_orderpriority
            ORDER BY o_orderpriority;
        """,
        "instances": [
            {"date": "1993-07-01"},
            {"date": "1994-01-01"},
        ],
    },
    5: {
        "name": "Local Supplier Volume",
        "sql": """
            SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue
            FROM customer, orders, lineitem, supplier, nation, region
            WHERE c_custkey = o_custkey
              AND l_orderkey = o_orderkey
              AND l_suppkey = s_suppkey
              AND c_nationkey = s_nationkey
              AND s_nationkey = n_nationkey
              AND n_regionkey = r_regionkey
              AND r_name = '{region}'
              AND o_orderdate >= DATE '{date}'
              AND o_orderdate < DATE '{date}' + INTERVAL '1 year'
            GROUP BY n_name
            ORDER BY revenue DESC;
        """,
        "instances": [
            {"region": "ASIA", "date": "1994-01-01"},
            {"region": "EUROPE", "date": "1995-01-01"},
        ],
    },
    6: {
        "name": "Forecasting Revenue",
        "sql": """
            SELECT SUM(l_extendedprice * l_discount) AS revenue
            FROM lineitem
            WHERE l_shipdate >= DATE '{date}'
              AND l_shipdate < DATE '{date}' + INTERVAL '1 year'
              AND l_discount BETWEEN {discount} - 0.01 AND {discount} + 0.01
              AND l_quantity < {quantity};
        """,
        "instances": [
            {"date": "1994-01-01", "discount": 0.06, "quantity": 24},
            {"date": "1995-01-01", "discount": 0.08, "quantity": 25},
        ],
    },
    7: {
        "name": "Volume Shipping",
        "sql": """
            SELECT supp_nation, cust_nation, l_year, SUM(volume) AS revenue
            FROM (
                SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,
                       EXTRACT(YEAR FROM l_shipdate) AS l_year,
                       l_extendedprice * (1 - l_discount) AS volume
                FROM supplier, lineitem, orders, customer, nation n1, nation n2
                WHERE s_suppkey = l_suppkey
                  AND o_orderkey = l_orderkey
                  AND c_custkey = o_custkey
                  AND s_nationkey = n1.n_nationkey
                  AND c_nationkey = n2.n_nationkey
                  AND (
                      (n1.n_name = '{nation1}' AND n2.n_name = '{nation2}')
                      OR (n1.n_name = '{nation2}' AND n2.n_name = '{nation1}')
                  )
                  AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
            ) AS shipping
            GROUP BY supp_nation, cust_nation, l_year
            ORDER BY supp_nation, cust_nation, l_year;
        """,
        "instances": [
            {"nation1": "FRANCE", "nation2": "GERMANY"},
            {"nation1": "JAPAN", "nation2": "INDIA"},
        ],
    },
    8: {
        "name": "National Market Share",
        "sql": """
            SELECT o_year, SUM(CASE WHEN nation = '{nation}' THEN volume ELSE 0 END) / SUM(volume) AS mkt_share
            FROM (
                SELECT EXTRACT(YEAR FROM o_orderdate) AS o_year,
                       l_extendedprice * (1 - l_discount) AS volume,
                       n2.n_name AS nation
                FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
                WHERE p_partkey = l_partkey
                  AND s_suppkey = l_suppkey
                  AND l_orderkey = o_orderkey
                  AND o_custkey = c_custkey
                  AND c_nationkey = n1.n_nationkey
                  AND n1.n_regionkey = r_regionkey
                  AND r_name = '{region}'
                  AND s_nationkey = n2.n_nationkey
                  AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
                  AND p_type = '{type}'
            ) AS all_nations
            GROUP BY o_year
            ORDER BY o_year;
        """,
        "instances": [
            {"nation": "BRAZIL", "region": "AMERICA", "type": "ECONOMY ANODIZED STEEL"},
            {"nation": "INDIA", "region": "ASIA", "type": "SMALL BRUSHED BRASS"},
        ],
    },
    9: {
        "name": "Product Type Profit Measure",
        "sql": """
            SELECT nation, o_year, SUM(amount) AS sum_profit
            FROM (
                SELECT n_name AS nation, EXTRACT(YEAR FROM o_orderdate) AS o_year,
                       l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
                FROM part, supplier, lineitem, partsupp, orders, nation
                WHERE s_suppkey = l_suppkey
                  AND ps_suppkey = l_suppkey
                  AND ps_partkey = l_partkey
                  AND p_partkey = l_partkey
                  AND o_orderkey = l_orderkey
                  AND s_nationkey = n_nationkey
                  AND p_name LIKE '%{color}%'
            ) AS profit
            GROUP BY nation, o_year
            ORDER BY nation, o_year DESC;
        """,
        "instances": [
            {"color": "green"},
            {"color": "red"},
        ],
    },
    10: {
        "name": "Returned Item Reporting",
        "sql": """
            SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue,
                   c_acctbal, n_name, c_address, c_phone, c_comment
            FROM customer, orders, lineitem, nation
            WHERE c_custkey = o_custkey
              AND l_orderkey = o_orderkey
              AND o_orderdate >= DATE '{date}'
              AND o_orderdate < DATE '{date}' + INTERVAL '3 month'
              AND l_returnflag = 'R'
              AND c_nationkey = n_nationkey
            GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
            ORDER BY revenue DESC
            LIMIT 20;
        """,
        "instances": [
            {"date": "1993-10-01"},
            {"date": "1994-03-01"},
        ],
    },
    12: {
        "name": "Shipping Modes and Order Priority",
        "sql": """
            SELECT l_shipmode,
                   SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
                   SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
            FROM orders, lineitem
            WHERE o_orderkey = l_orderkey
              AND l_shipmode IN ('{shipmode1}', '{shipmode2}')
              AND l_commitdate < l_receiptdate
              AND l_shipdate < l_commitdate
              AND l_receiptdate >= DATE '{date}'
              AND l_receiptdate < DATE '{date}' + INTERVAL '1 year'
            GROUP BY l_shipmode
            ORDER BY l_shipmode;
        """,
        "instances": [
            {"shipmode1": "MAIL", "shipmode2": "SHIP", "date": "1994-01-01"},
            {"shipmode1": "AIR", "shipmode2": "REG AIR", "date": "1995-01-01"},
        ],
    },
    14: {
        "name": "Promotion Effect",
        "sql": """
            SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) /
                   SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
            FROM lineitem, part
            WHERE l_partkey = p_partkey
              AND l_shipdate >= DATE '{date}'
              AND l_shipdate < DATE '{date}' + INTERVAL '1 month';
        """,
        "instances": [
            {"date": "1995-09-01"},
            {"date": "1996-04-01"},
        ],
    },
    19: {
        "name": "Discounted Revenue",
        "sql": """
            SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue
            FROM lineitem, part
            WHERE (
                p_partkey = l_partkey
                AND p_brand = '{brand1}'
                AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                AND l_quantity >= 1 AND l_quantity <= 11
                AND p_size BETWEEN 1 AND 5
                AND l_shipmode IN ('AIR', 'AIR REG')
                AND l_shipinstruct = 'DELIVER IN PERSON'
            ) OR (
                p_partkey = l_partkey
                AND p_brand = '{brand2}'
                AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                AND l_quantity >= 10 AND l_quantity <= 20
                AND p_size BETWEEN 1 AND 10
                AND l_shipmode IN ('AIR', 'AIR REG')
                AND l_shipinstruct = 'DELIVER IN PERSON'
            ) OR (
                p_partkey = l_partkey
                AND p_brand = '{brand3}'
                AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                AND l_quantity >= 20 AND l_quantity <= 30
                AND p_size BETWEEN 1 AND 15
                AND l_shipmode IN ('AIR', 'AIR REG')
                AND l_shipinstruct = 'DELIVER IN PERSON'
            );
        """,
        "instances": [
            {"brand1": "Brand#12", "brand2": "Brand#23", "brand3": "Brand#34"},
            {"brand1": "Brand#21", "brand2": "Brand#32", "brand3": "Brand#43"},
        ],
    },
}

ALL_HINTS = [
    ("default", {}),
    ("no_hash", {"enable_hashjoin": "off"}),
    ("no_merge", {"enable_mergejoin": "off"}),
    ("no_nestloop", {"enable_nestloop": "off"}),
    ("only_nestloop", {"enable_hashjoin": "off", "enable_mergejoin": "off"}),
    ("only_merge", {"enable_hashjoin": "off", "enable_nestloop": "off"}),
    ("only_hash", {"enable_mergejoin": "off", "enable_nestloop": "off"}),
    ("no_index", {"enable_indexscan": "off", "enable_bitmapscan": "off"}),
    ("no_seq", {"enable_seqscan": "off"}),
    ("idx_nestloop", {"enable_seqscan": "off", "enable_hashjoin": "off", "enable_mergejoin": "off"}),
    ("seq_merge", {"enable_indexscan": "off", "enable_bitmapscan": "off", "enable_hashjoin": "off", "enable_nestloop": "off"}),
    ("idx_hash", {"enable_seqscan": "off", "enable_mergejoin": "off", "enable_nestloop": "off"}),
]

QUICK_HINTS = [
    ("default", {}),
    ("no_hash", {"enable_hashjoin": "off"}),
    ("no_merge", {"enable_mergejoin": "off"}),
    ("only_nestloop", {"enable_hashjoin": "off", "enable_mergejoin": "off"}),
    ("no_index", {"enable_indexscan": "off", "enable_bitmapscan": "off"}),
    ("no_seq", {"enable_seqscan": "off"}),
]


def clean_plan_for_model(plan_node: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively remove runtime metrics from ANALYZE so only plain EXPLAIN features exist.
    """
    cleaned = {}
    for k, v in plan_node.items():
        if k in ("Actual Rows", "Actual Total Time", "Actual Loops", "Actual Startup Time"):
            continue
        if k == "Plans" and isinstance(v, list):
            cleaned["Plans"] = [clean_plan_for_model(child) for child in v]
        else:
            cleaned[k] = v
    return cleaned


def plan_signature(plan_node: dict[str, Any]) -> tuple:
    """
    Compute structural signature of plan AST to identify duplicate physical trees.
    """
    node_type = plan_node.get("Node Type", "")
    relation = plan_node.get("Relation Name", "")
    index = plan_node.get("Index Name", "")
    join_type = plan_node.get("Join Type", "")
    children = tuple(plan_signature(c) for c in plan_node.get("Plans", []))
    return (node_type, relation, index, join_type, children)


def extract_plans(
    quick: bool = False,
    host: str = "localhost",
    port: int = 5432,
    timeout_ms: int = 4000,
) -> dict[str, Any]:
    t_start = time.time()
    out_dir = Path("data/plans_real")
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("*.json"):
        existing.unlink()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    hints = QUICK_HINTS if quick else ALL_HINTS
    template_ids = [2, 3, 5, 7, 10] if quick else sorted(TEMPLATES.keys())

    logger.info("Connecting to PostgreSQL at %s:%d (Quick mode: %s)...", host, port, quick)
    con = psycopg2.connect(host=host, port=port, dbname="tpch", user="postgres", password="postgres")
    con.autocommit = True
    cur = con.cursor()

    # Query PG settings for documentation
    cur.execute("SHOW server_version;")
    pg_version = cur.fetchone()[0]
    cur.execute("SHOW shared_buffers;")
    shared_buffers = cur.fetchone()[0]
    cur.execute("SHOW work_mem;")
    work_mem = cur.fetchone()[0]

    logger.info("PostgreSQL Settings: version=%s, shared_buffers=%s, work_mem=%s", pg_version, shared_buffers, work_mem)

    total_query_instances = 0
    total_plans_saved = 0
    total_timed_out = 0
    plans_per_template: dict[int, int] = {}
    distinct_plans_per_query: dict[str, int] = {}

    for t_id in template_ids:
        t_info = TEMPLATES[t_id]
        t_name = t_info["name"]
        raw_sql = t_info["sql"]
        instances = t_info["instances"][:1] if quick else t_info["instances"]

        plans_per_template[t_id] = 0

        for inst_idx, params in enumerate(instances):
            query_id = f"t{t_id}_i{inst_idx}"
            sql = raw_sql.format(**params)
            total_query_instances += 1

            seen_signatures: dict[tuple, dict[str, Any]] = {}
            plan_counter = 0

            for hint_name, gucs in hints:
                is_default = (hint_name == "default")

                # Set session GUCs
                cur.execute("RESET ALL;")
                cur.execute("SET max_parallel_workers_per_gather = 0;")
                cur.execute(f"SET statement_timeout = {timeout_ms};")
                for k, v in gucs.items():
                    cur.execute(f"SET {k} = {v};")

                # 1. Plain EXPLAIN to obtain structure & check duplicates
                try:
                    cur.execute(f"EXPLAIN (FORMAT JSON) {sql};")
                    plain_res = cur.fetchone()[0]
                    plain_plan = plain_res[0]["Plan"]
                except Exception as e:
                    logger.warning("Plain EXPLAIN failed for %s hint %s: %s", query_id, hint_name, e)
                    continue

                sig = plan_signature(plain_plan)
                if sig in seen_signatures:
                    # Duplicate physical plan structure already captured for this query instance
                    continue

                # 2. Timing execution: warm-up run + 3 measured runs
                timed_out = False
                measured_times = []

                # Cold / Warm-up run
                try:
                    cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql};")
                    _ = cur.fetchone()[0]
                except psycopg2.errors.QueryCanceled:
                    timed_out = True
                    con.rollback()
                except Exception as e:
                    logger.warning("Warm-up run failed for %s hint %s: %s", query_id, hint_name, e)
                    timed_out = True

                if not timed_out:
                    for _ in range(3):
                        try:
                            cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql};")
                            analyze_res = cur.fetchone()[0]
                            exec_time = float(analyze_res[0]["Execution Time"])
                            measured_times.append(exec_time)
                        except psycopg2.errors.QueryCanceled:
                            timed_out = True
                            con.rollback()
                            break
                        except Exception as e:
                            logger.warning("Measured run failed for %s hint %s: %s", query_id, hint_name, e)
                            timed_out = True
                            break

                if timed_out or not measured_times:
                    final_runtime_ms = float(timeout_ms)
                    timed_out = True
                    total_timed_out += 1
                else:
                    measured_times.sort()
                    final_runtime_ms = measured_times[1]  # Median of 3

                cleaned_plan = clean_plan_for_model(plain_plan)

                record = {
                    "query_id": query_id,
                    "template_id": t_id,
                    "template_name": t_name,
                    "plan_id": plan_counter,
                    "hint_name": hint_name,
                    "is_default_plan": is_default,
                    "execution_time": round(final_runtime_ms, 3),
                    "timed_out": timed_out,
                    "measurements_median_of": 3,
                    "timeout_cap_ms": float(timeout_ms),
                    "plan": cleaned_plan,
                }

                seen_signatures[sig] = record
                plan_file = out_dir / f"plan_{query_id}_p{plan_counter:02d}_{hint_name}.json"
                with open(plan_file, "w") as fp:
                    json.dump(record, fp, indent=2)

                plan_counter += 1
                total_plans_saved += 1
                plans_per_template[t_id] += 1

            distinct_plans_per_query[query_id] = plan_counter
            logger.info("Query %s (%s): %d distinct plan variants extracted.", query_id, t_name, plan_counter)

    cur.close()
    con.close()
    elapsed = time.time() - t_start

    # Check for templates with fewer than 2 distinct plans
    excluded_templates = [t for t, count in plans_per_template.items() if count < 2]

    summary = {
        "postgres_version": pg_version,
        "shared_buffers": shared_buffers,
        "work_mem": work_mem,
        "max_parallel_workers_per_gather": 0,
        "statement_timeout_ms": timeout_ms,
        "total_query_instances": total_query_instances,
        "total_plans_saved": total_plans_saved,
        "total_timed_out": total_timed_out,
        "timeout_fraction": round(total_timed_out / max(1, total_plans_saved), 4),
        "plans_per_template": plans_per_template,
        "distinct_plans_per_query": distinct_plans_per_query,
        "excluded_templates_under_2_plans": excluded_templates,
        "elapsed_seconds": round(elapsed, 2),
    }

    with open(results_dir / "real_data_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Extracted %d plans across %d query instances in %.2fs. Timed out: %d (%.1f%%).",
                total_plans_saved, total_query_instances, elapsed, total_timed_out, summary["timeout_fraction"] * 100)
    if excluded_templates:
        logger.warning("Excluded templates with < 2 distinct plans: %s", excluded_templates)

    return summary


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Extract real PostgreSQL TPC-H plans and execution timings.")
    parser.add_argument("--quick", action="store_true", help="Quick mode for fast development (< 2 minutes)")
    parser.add_argument("--timeout", type=int, default=4000, help="Statement timeout in ms")
    args = parser.parse_args()

    extract_plans(quick=args.quick, timeout_ms=args.timeout)


if __name__ == "__main__":
    main()
