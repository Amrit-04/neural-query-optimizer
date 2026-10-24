"""
Setup TPC-H Schema and Data in PostgreSQL using DuckDB.

Scale factor: SF 0.05 (~300,000 lineitem rows).
Generates in ~3s, copies directly to PostgreSQL via COPY, builds B-tree indexes, and runs ANALYZE.
Logs exact row counts for audit verification.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import psycopg2

from src.logging_config import setup_logging

logger = logging.getLogger("neural_optimizer")

SCHEMA_SQL = """
DROP TABLE IF EXISTS lineitem CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS partsupp CASCADE;
DROP TABLE IF EXISTS customer CASCADE;
DROP TABLE IF EXISTS supplier CASCADE;
DROP TABLE IF EXISTS part CASCADE;
DROP TABLE IF EXISTS nation CASCADE;
DROP TABLE IF EXISTS region CASCADE;

CREATE TABLE region (
    r_regionkey INTEGER PRIMARY KEY,
    r_name CHAR(25) NOT NULL,
    r_comment VARCHAR(152)
);

CREATE TABLE nation (
    n_nationkey INTEGER PRIMARY KEY,
    n_name CHAR(25) NOT NULL,
    n_regionkey INTEGER NOT NULL REFERENCES region(r_regionkey),
    n_comment VARCHAR(152)
);

CREATE TABLE part (
    p_partkey INTEGER PRIMARY KEY,
    p_name VARCHAR(55) NOT NULL,
    p_mfgr CHAR(25) NOT NULL,
    p_brand CHAR(10) NOT NULL,
    p_type VARCHAR(25) NOT NULL,
    p_size INTEGER NOT NULL,
    p_container CHAR(10) NOT NULL,
    p_retailprice DECIMAL(15,2) NOT NULL,
    p_comment VARCHAR(23) NOT NULL
);

CREATE TABLE supplier (
    s_suppkey INTEGER PRIMARY KEY,
    s_name CHAR(25) NOT NULL,
    s_address VARCHAR(40) NOT NULL,
    s_nationkey INTEGER NOT NULL REFERENCES nation(n_nationkey),
    s_phone CHAR(15) NOT NULL,
    s_acctbal DECIMAL(15,2) NOT NULL,
    s_comment VARCHAR(101) NOT NULL
);

CREATE TABLE partsupp (
    ps_partkey INTEGER NOT NULL REFERENCES part(p_partkey),
    ps_suppkey INTEGER NOT NULL REFERENCES supplier(s_suppkey),
    ps_availqty INTEGER NOT NULL,
    ps_supplycost DECIMAL(15,2) NOT NULL,
    ps_comment VARCHAR(199) NOT NULL,
    PRIMARY KEY (ps_partkey, ps_suppkey)
);

CREATE TABLE customer (
    c_custkey INTEGER PRIMARY KEY,
    c_name VARCHAR(25) NOT NULL,
    c_address VARCHAR(40) NOT NULL,
    c_nationkey INTEGER NOT NULL REFERENCES nation(n_nationkey),
    c_phone CHAR(15) NOT NULL,
    c_acctbal DECIMAL(15,2) NOT NULL,
    c_mktsegment CHAR(10) NOT NULL,
    c_comment VARCHAR(117) NOT NULL
);

CREATE TABLE orders (
    o_orderkey INTEGER PRIMARY KEY,
    o_custkey INTEGER NOT NULL REFERENCES customer(c_custkey),
    o_orderstatus CHAR(1) NOT NULL,
    o_totalprice DECIMAL(15,2) NOT NULL,
    o_orderdate DATE NOT NULL,
    o_orderpriority CHAR(15) NOT NULL,
    o_clerk CHAR(15) NOT NULL,
    o_shippriority INTEGER NOT NULL,
    o_comment VARCHAR(79) NOT NULL
);

CREATE TABLE lineitem (
    l_orderkey INTEGER NOT NULL REFERENCES orders(o_orderkey),
    l_partkey INTEGER NOT NULL REFERENCES part(p_partkey),
    l_suppkey INTEGER NOT NULL REFERENCES supplier(s_suppkey),
    l_linenumber INTEGER NOT NULL,
    l_quantity DECIMAL(15,2) NOT NULL,
    l_extendedprice DECIMAL(15,2) NOT NULL,
    l_discount DECIMAL(15,2) NOT NULL,
    l_tax DECIMAL(15,2) NOT NULL,
    l_returnflag CHAR(1) NOT NULL,
    l_linestatus CHAR(1) NOT NULL,
    l_shipdate DATE NOT NULL,
    l_commitdate DATE NOT NULL,
    l_receiptdate DATE NOT NULL,
    l_shipinstruct CHAR(25) NOT NULL,
    l_shipmode CHAR(10) NOT NULL,
    l_comment VARCHAR(44) NOT NULL,
    PRIMARY KEY (l_orderkey, l_linenumber)
);
"""

INDEX_SQL = """
CREATE INDEX idx_orders_custkey ON orders(o_custkey);
CREATE INDEX idx_orders_orderdate ON orders(o_orderdate);
CREATE INDEX idx_lineitem_orderkey ON lineitem(l_orderkey);
CREATE INDEX idx_lineitem_partkey ON lineitem(l_partkey);
CREATE INDEX idx_lineitem_suppkey ON lineitem(l_suppkey);
CREATE INDEX idx_lineitem_shipdate ON lineitem(l_shipdate);
CREATE INDEX idx_customer_nationkey ON customer(c_nationkey);
CREATE INDEX idx_supplier_nationkey ON supplier(s_nationkey);
CREATE INDEX idx_partsupp_partkey ON partsupp(ps_partkey);
CREATE INDEX idx_partsupp_suppkey ON partsupp(ps_suppkey);
"""

TABLES_ORDER = ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]


def setup_database(scale_factor: float = 0.05, host: str = "localhost", port: int = 5432) -> dict[str, int]:
    t0 = time.time()
    logger.info("Initializing DuckDB TPC-H generation at Scale Factor %.2f...", scale_factor)
    duck_con = duckdb.connect()
    duck_con.execute(f"CALL dbgen(sf={scale_factor});")
    logger.info("DuckDB generation complete in %.2fs", time.time() - t0)

    logger.info("Connecting to PostgreSQL at %s:%d...", host, port)
    pg_con = psycopg2.connect(host=host, port=port, dbname="tpch", user="postgres", password="postgres")
    pg_con.autocommit = True
    cur = pg_con.cursor()
    cur.execute("SET statement_timeout = 0;")

    logger.info("Creating TPC-H relational schema...")
    cur.execute(SCHEMA_SQL)

    row_counts = {}
    for tbl in TABLES_ORDER:
        t_copy = time.time()
        # Export DuckDB table directly to CSV in-memory buffer
        df = duck_con.execute(f"SELECT * FROM {tbl};").df()
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N")
        buf.seek(0)

        cur.copy_from(buf, tbl, sep="\t", null="\\N")
        cur.execute(f"SELECT COUNT(*) FROM {tbl};")
        cnt = cur.fetchone()[0]
        row_counts[tbl] = cnt
        logger.info("Loaded table %s: %d rows (%.2fs)", tbl, cnt, time.time() - t_copy)

    logger.info("Building B-Tree indexes on join and filter columns...")
    t_idx = time.time()
    cur.execute(INDEX_SQL)
    logger.info("Indexes built in %.2fs", time.time() - t_idx)

    logger.info("Running ANALYZE to update PostgreSQL optimizer statistics...")
    t_ana = time.time()
    cur.execute("ANALYZE;")
    logger.info("ANALYZE complete in %.2fs", time.time() - t_ana)

    cur.close()
    pg_con.close()
    elapsed = time.time() - t0
    logger.info("TPC-H PostgreSQL Setup Finished Successfully in %.2fs!", elapsed)
    return row_counts


def main():
    setup_logging()
    counts = setup_database(scale_factor=0.05)
    print("\n=== Verified TPC-H Database Row Counts ===")
    for t, c in counts.items():
        print(f"  - {t}: {c:,} rows")


if __name__ == "__main__":
    main()
