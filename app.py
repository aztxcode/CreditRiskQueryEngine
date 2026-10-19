"""
Northbridge Bank - Credit Risk Natural-Language Query Engine
Streamlit deployment of the Project 3 notebook pipeline.

This app preserves the original notebook logic:
- Verified Query Template Library (VQ1-VQ10)
- Intent classification (verified vs. generated route)
- SQL generation for novel questions (read-only, SQLite-compatible)
- Five-check SQL validation gate
- One retry on validation failure (generated track only)
- Escalation to human analyst if retry also fails
- Query execution against a read-only SQLite connection
- Business-focused narrative response generation
- Full audit trail (route, query id, SQL, confidence, warnings) for every answer

UI is organized into three tabs: Ask a Question, Test Cases & Evaluation, and Audit Trail.
"""

import os
import re
import json
import sqlite3
import warnings

import pandas as pd
import numpy as np
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Northbridge Bank | Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

# --------------------------------------------------------------------------------------
# Configuration: API credentials, model names, database path
# --------------------------------------------------------------------------------------
# Credentials are read from Streamlit secrets first, then from environment variables.
# Configure these in .streamlit/secrets.toml (local) or the app's "Secrets" panel
# (Streamlit Community Cloud) - never hard-code keys in the source file.
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
OPENAI_API_BASE = st.secrets.get("OPENAI_BASE_URL", os.environ.get("OPENAI_BASE_URL", ""))

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE

GENERATOR_MODEL = st.secrets.get("GENERATOR_MODEL", os.environ.get("GENERATOR_MODEL", "gpt-4o-mini"))
EVALUATOR_MODEL = st.secrets.get("EVALUATOR_MODEL", os.environ.get("EVALUATOR_MODEL", "gpt-4o"))

# The database file is expected to sit alongside app.py in the deployed repo.
# Override with the DB_PATH secret/env var if it lives elsewhere.
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_risk_portfolio.db")
DB_PATH = st.secrets.get("DB_PATH", os.environ.get("DB_PATH", DEFAULT_DB_PATH))

# test_queries.csv is expected to sit alongside app.py as well, but can also be uploaded
# from the "Test Cases & Evaluation" tab if it lives elsewhere.
DEFAULT_TEST_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_queries.csv")
TEST_CSV_PATH = st.secrets.get("TEST_CSV_PATH", os.environ.get("TEST_CSV_PATH", DEFAULT_TEST_CSV_PATH))

# --------------------------------------------------------------------------------------
# Database schema context (provided to the LLM in every prompt) - unchanged from notebook
# --------------------------------------------------------------------------------------
DATABASE_SCHEMA = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# --------------------------------------------------------------------------------------
# Resource initialization (cached so the app does not reconnect / re-instantiate on every rerun)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llms():
    llm = ChatOpenAI(model=GENERATOR_MODEL, temperature=0)
    evaluator_llm = ChatOpenAI(model=EVALUATOR_MODEL, temperature=0)
    return llm, evaluator_llm


@st.cache_resource(show_spinner=False)
def get_connection(db_path):
    # Read-only connection via URI mode to prevent any write operations.
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    return conn


@st.cache_data(show_spinner=False)
def build_verified_query_library():
    sql_1 = """
SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_exposure_million,
    ROUND(SUM(CASE
        WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN l.total_outstanding
        ELSE 0
    END) / 1000000.0, 2) AS npa_exposure_million
FROM loan_master l
JOIN sector_master s
    ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_exposure_million DESC
"""

    sql_2 = """SELECT
    loan_category,
    ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_million,
    COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_million DESC"""

    sql_3 = """SELECT
    ifrs9_stage,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(ead_amount) / 1e6, 2) AS total_ead_million,
    ROUND(SUM(ecl_amount) / 1e6, 2) AS total_ecl_million
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage ASC"""

    sql_4 = """SELECT
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio_pct
FROM provisioning p
JOIN loan_master l
    ON p.loan_account_number = l.loan_account_number
JOIN sector_master s
    ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio_pct DESC"""

    sql_5 = """SELECT
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1e6, 2) AS total_outstanding_million,
    l.asset_classification
FROM loan_master l
JOIN sector_master s
    ON l.sector_code = s.sector_code
ORDER BY total_outstanding_million DESC
LIMIT 10"""

    sql_6 = """SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_million
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_million DESC
LIMIT 5"""

    sql_7 = """SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1e6, 2) AS total_outstanding_million,
    l.days_past_due,
    l.asset_classification
FROM loan_master l
JOIN sector_master s
    ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC"""

    sql_8 = """SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_million
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 1
        WHEN '1-30' THEN 2
        WHEN '31-60' THEN 3
        WHEN '61-90' THEN 4
        WHEN '90+' THEN 5
    END ASC"""

    sql_9 = """SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC"""

    sql_10 = """SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1e6, 2) AS total_ecl_million
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC"""

    verified_query_library = {
        "VQ1": {
            "description": "Sector-wise total outstanding and NPA amount breakdown across all sectors",
            "sql": sql_1,
        },
        "VQ2": {
            "description": "Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)",
            "sql": sql_2,
        },
        "VQ3": {
            "description": "IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter",
            "sql": sql_3,
        },
        "VQ4": {
            "description": "Average provision coverage ratio by sector for the latest reporting quarter",
            "sql": sql_4,
        },
        "VQ5": {
            "description": "Top 10 largest loan exposures by outstanding amount at the borrower level",
            "sql": sql_5,
        },
        "VQ6": {
            "description": "Top 5 largest exposures aggregated at the business group level",
            "sql": sql_6,
        },
        "VQ7": {
            "description": "All overdue loan accounts with their days past due and asset classification",
            "sql": sql_7,
        },
        "VQ8": {
            "description": "Distribution of loans across days-past-due buckets showing aging profile of the portfolio",
            "sql": sql_8,
        },
        "VQ9": {
            "description": "Borrowers whose internal rating was downgraded in the latest rating cycle",
            "sql": sql_9,
        },
        "VQ10": {
            "description": "Expected credit loss trend across all reporting quarters showing provisioning movement over time",
            "sql": sql_10,
        },
    }
    return verified_query_library


# --------------------------------------------------------------------------------------
# Pipeline tools (ported verbatim in logic from the notebook)
# --------------------------------------------------------------------------------------
def classify_intent(user_question, query_library, llm):
    """
    Classifies the user question and decides which route to take.

    Returns:
    - dict: Contains 'route' (verified or generated),
            'query_id' (template ID or None),
            'match_reason' (short explanation of the decision).
    """
    library_descriptions = "\n".join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

### ROLE
You are a query router for a financial operations analytics system. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording.
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about shape of answer: a question asking for row-level detail should NOT match a template that returns an aggregate count.

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context, llm):
    """
    Generates a candidate SQL query for a novel question using the database schema.
    """
    generation_prompt = f"""

### ROLE
You are a senior SQL developer specializing in financial operations analytics on a SQLite database.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. Resolve named entities using appropriate master/attribute columns.
5. Ensure the query is SQLite compatible.
6. For monetary amounts (e.g., total_outstanding, ead_amount, ecl_amount): use ROUND(.../ 1e6, 2) to convert to millions.
7. NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
8. For the latest provisioning data, use: reporting_date = '2025-09-30'
9. For the latest rating data, use: rating_date = '2025-09-30'
10. In SQLite, never subtract DATE() or date columns directly (e.g. DATE(a)-DATE(b)) — it silently returns 0; always use julianday(a)-julianday(b) for day differences.
11. Alias every numeric column with a suffix that states its unit, so the result is self-describing. Use _usd for dollar amounts, _pct or _percent for percentages, _count for counts, _hours for hour values, and _days for day values. Avoid bare aliases like "value", "amount", or "total".

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.

"""

    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r"^```\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, llm, evaluator_llm, query_id=None):
    """
    Validates a candidate SQL query through five checks before execution.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (float, 0-1).
    """
    result = {
        "passed": False,
        "failed_check": None,
        "details": "",
        "relevance_confidence": None,
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "ATTACH"]
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH"
        return result
    for kw in forbidden_keywords:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            result["failed_check"] = "read_only_shape"
            result["details"] = f"Forbidden keyword detected: {kw}"
            return result

    # Robust check for multiple statements using sqlparse
    parsed_statements = sqlparse.parse(candidate_sql)
    non_empty_statements = [s for s in parsed_statements if str(s).strip()]
    if len(non_empty_statements) > 1:
        result["failed_check"] = "read_only_shape"
        result["details"] = "Multiple statements are not allowed"
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    referenced_identifiers = re.findall(r"\b[a-z_][a-z0-9_]*\b", candidate_sql.lower())
    sql_keywords = {
        "select", "from", "where", "and", "or", "group", "by", "order", "having", "limit", "join", "on", "as", "case",
        "when", "then", "else", "end", "sum", "count", "avg", "min", "max", "round", "desc", "asc", "left", "right",
        "inner", "outer", "distinct", "null", "is", "not", "in", "like", "with", "union", "all", "between", "coalesce",
    }
    unknown = [
        tok for tok in referenced_identifiers
        if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
        and not tok.isdigit() and tok not in ("s", "l", "p", "r", "e6")
    ]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result["failed_check"] = "parse_plan_dry_run"
        result["details"] = f"SQL failed to parse or plan: {str(e)}"
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""

### ROLE
You are a SQL validation expert for a commercial bank's credit risk system. Evaluate whether the SQL query correctly answers the user's qustion.

### CONTEXT
{track_context}

### INPUT
User Question:
{user_question}

### SQL QUERY TO EVALUATE
{candidate_sql}

### INSTRUCTIONS AND EVALUATION CRITERIA
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does the SQL query use the correct tables for this question?
2. Does it calculate the right metrics (outstanding, NPA, ECL, interest rate, etc.)?
3. Does it apply appropriate filters (NPA definition, date filters, etc.) whee needed?
4. Is the aggregation and grouping logic correct?

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r"\{.*\}", relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result["relevance_confidence"] = relevance_json.get("confidence", 0.0)
        if relevance_json.get("verdict") == "no" or relevance_json.get("confidence", 0.0) < 0.6:
            result["failed_check"] = "llm_relevance"
            result["details"] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]["sql"]
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result["failed_check"] = "template_integrity"
                result["details"] = f"Expected {len(expected_cols)} columns, got {len(actual_cols)}"
                return result
        except sqlite3.Error as e:
            result["failed_check"] = "template_integrity"
            result["details"] = f"Template integrity check failed: {str(e)}"
            return result

    result["passed"] = True
    result["details"] = "All validation checks passed"
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context, llm):
    """
    Regenerates SQL after a validation failure, feeding the error back to the LLM.
    """
    retry_prompt = f"""
### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.

"""
    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r"^```sql\s*|\s*```$", "", revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r"^```\s*|\s*```$", "", revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    """
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    """
    result = {"dataframe": None, "reasonable": True, "warnings": []}

    df = pd.read_sql_query(validated_sql, db_connection)
    result["dataframe"] = df

    if df.empty:
        result["warnings"].append("Query returned an empty result")

    for col in df.select_dtypes(include="number").columns:
        if (df[col] < 0).any() and "deviation" not in col.lower() and "change" not in col.lower():
            result["warnings"].append(f"Column {col} contains negative values")
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result["warnings"].append(f"Column {col} has {null_count} null values")

    if len(result["warnings"]) > 2:
        result["reasonable"] = False

    return result


def generate_response(user_question, dataframe, route, llm, query_id=None):
    """
    Generates a focused natural language response from the query result.
    """
    response_prompt = f"""

You are a commercial credit risk analytics assistant for Northbridge Bank.
Your task is to generate a concise, professional, and business-focused response to the user's question using the provided query result data.

### CONTEXT & RULES:
1. Focus directly on answering what the user specifically asked.
2. If the query came from a verified template ({route}), the table may contain portfolio-wide data (e.g., all sectors or buckets). Locate and highlight only the specific entity, sector, or metric requested by the user, while providing concise context if helpful.
3. If the dataframe is empty, state clearly that no records matched the criteria.
4. Use exact numerical values from the table (cite figures in millions USD, percentages, loan counts, or days past due as indicated in the data). Do not invent numbers.
5. Keep the narrative concise (2 to 4 sentences or brief bullet points). Avoid fluff, pleasantries, or speculative explanations.

### USER QUESTION:
{user_question}

### QUERY RESULT DATA:
{dataframe.to_string(index=False)}

### RESPONSE:
Provide the executive summary narrative:

"""
    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# --------------------------------------------------------------------------------------
# Pipeline orchestration (ported from run_pipeline in the notebook)
# --------------------------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, llm, evaluator_llm, status_cb=None):
    """
    Runs the complete query engine pipeline for a single user question.

    status_cb: optional callable(str) used to stream stage-by-stage progress to the UI
               (replaces the notebook's verbose print statements).
    """

    def log_status(msg):
        if status_cb:
            status_cb(msg)

    log = {
        "user_question": user_question,
        "route": None,
        "query_id": None,
        "match_reason": None,
        "candidate_sql": None,
        "gate_result": None,
        "retry_used": False,
        "escalated": False,
        "executed_sql": None,
        "row_count": None,
        "confidence": None,
        "narrative": None,
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library, llm)
    log["route"] = classification["route"]
    log["query_id"] = classification.get("query_id")
    log["match_reason"] = classification.get("match_reason")
    log_status(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']} — {log['match_reason']}")

    # Step 2: Query construction
    if log["route"] == "verified" and log["query_id"] in query_library:
        candidate_sql = query_library[log["query_id"]]["sql"]
    else:
        candidate_sql = generate_query(user_question, schema_context, llm)
    log["candidate_sql"] = candidate_sql
    log_status(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, llm, evaluator_llm, log["query_id"])
    log["gate_result"] = gate
    log_status(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate["passed"]:
        log_status(f"    Failed check: {gate.get('failed_check')} — {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate["passed"] and log["route"] == "generated":
        log_status(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate["details"], schema_context, llm)
        log["candidate_sql"] = candidate_sql
        log["retry_used"] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, llm, evaluator_llm, None)
        log["gate_result"] = gate
        log_status(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate["passed"]:
            log_status(f"    Retry failed check: {gate.get('failed_check')} — {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate["passed"]:
        log["escalated"] = True
        log["narrative"] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log["confidence"] = "ESCALATED"
        log_status(f"[!] Escalated to human: {gate['details']}")
        return {"log": log, "dataframe": None, **log}

    # Step 6: Execute
    log["executed_sql"] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result["dataframe"]
    log["row_count"] = len(df)
    log["warnings"] = exec_result["warnings"]
    log_status(f"[4] Execute: {len(df)} rows returned" + (f" | Warnings: {exec_result['warnings']}" if exec_result["warnings"] else ""))

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log["route"], llm, log["query_id"])
    log["narrative"] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log["confidence"] = gate.get("relevance_confidence")
    log_status(f"[5] Response Generation: confidence={log['confidence']}")

    return {"log": log, "dataframe": df, **log}


# --------------------------------------------------------------------------------------
# Audit trail helpers
# --------------------------------------------------------------------------------------
def make_audit_entry(result):
    """
    Builds a JSON-serializable audit log entry from a run_pipeline() result.
    Captures routing decision, validation gate checks, retries, and execution details.
    """
    log = result["log"]
    df = result.get("dataframe")

    entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "user_question": log.get("user_question"),
        "routing_decision": {
            "route": log.get("route"),
            "query_id": log.get("query_id"),
            "match_reason": log.get("match_reason"),
        },
        "candidate_sql": log.get("candidate_sql"),
        "validation_gate": log.get("gate_result"),
        "retry_used": log.get("retry_used"),
        "escalated": log.get("escalated"),
        "execution": {
            "executed_sql": log.get("executed_sql"),
            "row_count": log.get("row_count"),
            "warnings": log.get("warnings", []),
        },
        "confidence": log.get("confidence"),
        "narrative": log.get("narrative"),
    }
    return entry


def append_to_audit_history(result):
    if "audit_history" not in st.session_state:
        st.session_state["audit_history"] = []
    st.session_state["audit_history"].append(make_audit_entry(result))


# --------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------
def render_result(result):
    """Render one pipeline result: status banner, narrative, SQL, data, confidence."""
    if result.get("escalated"):
        st.error("🚨 Escalated to human analyst")
        st.write(result["narrative"])
        with st.expander("Last attempted SQL"):
            st.code(result.get("candidate_sql", ""), language="sql")
        return

    route = result.get("route")
    query_id = result.get("query_id")
    confidence = result.get("confidence")

    badge = f"Route: **{route}**" + (f" (`{query_id}`)" if query_id else "")
    st.markdown(badge)

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence))
    col2.metric("Rows Returned", result.get("row_count"))
    col3.metric("Retry Used", "Yes" if result.get("retry_used") else "No")

    st.subheader("Narrative")
    st.write(result.get("narrative"))

    st.subheader("Executed SQL")
    st.code(result.get("executed_sql", ""), language="sql")

    st.subheader("Result Data")
    df = result.get("dataframe")
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data returned.")


def load_test_queries(uploaded_csv):
    """Load test_queries.csv from an uploaded file, else fall back to the bundled path."""
    if uploaded_csv is not None:
        return pd.read_csv(uploaded_csv)
    if os.path.exists(TEST_CSV_PATH):
        return pd.read_csv(TEST_CSV_PATH)
    return None


# --------------------------------------------------------------------------------------
# Tab 1: Ask a Question
# --------------------------------------------------------------------------------------
def render_ask_tab(conn, verified_query_library, llm, evaluator_llm):
    st.subheader("Ask a question")

    example_questions = [
        "What is the real estate sector's total outstanding and NPA exposure?",
        "Show me the DPD aging profile of the portfolio.",
        "How many restructured loans are impaired?",
        "What is the average interest rate by sector?",
        "How has Stage 3 exposure changed across reporting quarters?",
    ]
    with st.expander("Example questions"):
        for q in example_questions:
            st.markdown(f"- {q}")

    user_question = st.text_input(
        "Enter your portfolio question",
        placeholder="e.g., What is the sector-wise outstanding and NPA breakdown?",
        key="ask_tab_question",
    )
    show_trace = st.checkbox("Show pipeline trace (routing, validation, retries)", value=True, key="ask_tab_trace")
    ask_clicked = st.button("Ask", type="primary", key="ask_tab_button")

    if ask_clicked:
        if not user_question.strip():
            st.warning("Please enter a question first.")
            return

        trace_lines = []

        def status_cb(msg):
            trace_lines.append(msg)

        with st.spinner("Running the query engine pipeline..."):
            result = run_pipeline(
                user_question, conn, verified_query_library, DATABASE_SCHEMA, llm, evaluator_llm, status_cb=status_cb
            )

        # Every executed query appends its log to the session-state audit history.
        append_to_audit_history(result)

        if show_trace and trace_lines:
            with st.expander("Pipeline trace", expanded=False):
                for line in trace_lines:
                    st.text(line)

        st.markdown("---")
        render_result(result)


# --------------------------------------------------------------------------------------
# Tab 2: Test Cases & Evaluation
# --------------------------------------------------------------------------------------
def render_evaluation_tab(conn, verified_query_library, llm, evaluator_llm):
    st.subheader("Test Cases & Evaluation")
    st.write(
        "Runs the five benchmark test cases (TC-01 through TC-05) from `test_queries.csv` against the pipeline, "
        "then computes Path Accuracy, Query Accuracy, and Average Confidence against the ground truth."
    )

# Load test_queries.csv directly from repository
    try:
        ground_truth_preview = pd.read_csv("test_queries.csv")
        st.subheader("Benchmark Test Queries")
        st.dataframe(
            ground_truth_preview[["Test Case", "User Query", "Expected Route", "Expected Query ID", "Expected Answer"]],
            use_container_width=True
        )
    except Exception:
        pass

    run_clicked = st.button("Run 5 Benchmark Test Cases", type="primary", key="eval_tab_run")

    if run_clicked:
        try:
            ground_truth = pd.read_csv("test_queries.csv")
        except Exception:
            st.error("Could not find `test_queries.csv`. Please ensure it is uploaded in the root GitHub folder alongside `app.py`.")
            return

        # Restrict to the 5 benchmark cases TC-01 through TC-05, in that order, if a Test Case column exists.
        if "Test Case" in ground_truth.columns:
            benchmark_ids = [f"TC-0{i}" for i in range(1, 6)]
            subset = ground_truth[ground_truth["Test Case"].astype(str).isin(benchmark_ids)].copy()
            if not subset.empty:
                order = {tc: i for i, tc in enumerate(benchmark_ids)}
                subset["_order"] = subset["Test Case"].astype(str).map(order)
                ground_truth = subset.sort_values("_order").drop(columns="_order").reset_index(drop=True)
            else:
                ground_truth = ground_truth.head(5).reset_index(drop=True)
        else:
            ground_truth = ground_truth.head(5).reset_index(drop=True)

        evaluation_rows = []
        test_results = []
        progress = st.progress(0, text="Running test cases...")

        for i, (_, gt) in enumerate(ground_truth.iterrows()):
            result = run_pipeline(gt["User Query"], conn, verified_query_library, DATABASE_SCHEMA, llm, evaluator_llm)
            test_results.append(result)

            # Every executed query appends its log to the session-state audit history.
            append_to_audit_history(result)

            progress.progress((i + 1) / len(ground_truth), text=f"Running test cases... ({i + 1}/{len(ground_truth)})")

            tr = result["log"]
            evaluation_rows.append(
                {
                    "Test Case": gt.get("Test Case", f"TC-0{i + 1}"),
                    "User Query": gt.get("User Query"),
                    "Expected Route": gt.get("Expected Route"),
                    "Actual Route": tr["route"],
                    "Route Match": tr["route"] == gt.get("Expected Route"),
                    "Expected Query ID": gt.get("Expected Query ID"),
                    "Actual Query ID": tr["query_id"],
                    "Query ID Match": (
                        pd.isna(gt.get("Expected Query ID")) and pd.isna(tr["query_id"])
                    )
                    or tr["query_id"] == gt.get("Expected Query ID"),
                    "Confidence": tr["confidence"],
                    "Rows Returned": tr["row_count"],
                }
            )

        progress.empty()

        evaluation_df = pd.DataFrame(evaluation_rows)
        path_accuracy = evaluation_df["Route Match"].mean() * 100

        verified_mask = evaluation_df["Expected Route"].astype(str).str.strip().str.lower() == "verified"
        query_accuracy = (
            evaluation_df.loc[verified_mask, "Query ID Match"].mean() * 100 if verified_mask.any() else float("nan")
        )

        numeric_confidence = pd.to_numeric(evaluation_df["Confidence"], errors="coerce")
        average_confidence = numeric_confidence.mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Path Accuracy", f"{path_accuracy:.1f}%")
        c2.metric("Query Accuracy", f"{query_accuracy:.1f}%" if not pd.isna(query_accuracy) else "N/A")
        c3.metric("Average Confidence", f"{average_confidence:.2f}" if not pd.isna(average_confidence) else "N/A")

        st.subheader("Evaluation Results")
        st.dataframe(evaluation_df, use_container_width=True)

        st.session_state["last_evaluation_df"] = evaluation_df

        with st.expander("Full results per test case"):
            for i, result in enumerate(test_results):
                st.markdown(f"**{evaluation_rows[i]['Test Case']}: {evaluation_rows[i]['User Query']}**")
                render_result(result)
                st.markdown("---")
    elif "last_evaluation_df" in st.session_state:
        st.info("Showing results from the most recent run.")
        st.dataframe(st.session_state["last_evaluation_df"], use_container_width=True)


# --------------------------------------------------------------------------------------
# Tab 3: Audit Trail
# --------------------------------------------------------------------------------------
def render_audit_tab():
    st.subheader("Audit Trail")
    st.write(
        "Every question answered in this session — from the Ask tab or the Evaluation tab — is logged here, "
        "including the routing decision, validation gate checks, retries, and execution details."
    )

    history = st.session_state.get("audit_history", [])

    if not history:
        st.info("No queries have been run yet in this session.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{len(history)}** logged quer{'y' if len(history) == 1 else 'ies'} this session.")
    with col2:
        if st.button("Clear audit history", key="clear_audit_history"):
            st.session_state["audit_history"] = []
            st.rerun()

    download_payload = json.dumps(history, indent=2, default=str)
    st.download_button(
        "Download full audit trail (JSON)",
        data=download_payload,
        file_name="audit_trail.json",
        mime="application/json",
    )

    for i, entry in enumerate(reversed(history)):
        idx = len(history) - i
        route = entry["routing_decision"].get("route")
        query_id = entry["routing_decision"].get("query_id")
        escalated = entry.get("escalated")
        label_suffix = " — ESCALATED" if escalated else ""
        label = f"#{idx} [{entry.get('timestamp', '')}] {entry.get('user_question', '')} — route={route}" + (
            f" ({query_id})" if query_id else ""
        ) + label_suffix

        with st.expander(label):
            st.json(entry)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    st.title("🏦 Northbridge Bank — Credit Risk Query Engine")
    st.caption(
        "Ask routine commercial lending portfolio questions in plain English. "
        "Answers are produced via pre-approved verified templates where possible, "
        "or read-only generated SQL that is validated before execution — with a full audit trail every time."
    )

    if "audit_history" not in st.session_state:
        st.session_state["audit_history"] = []

    # ---- Preflight checks -------------------------------------------------------------
    if not OPENAI_API_KEY:
        st.error(
            "No OpenAI API key found. Add `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) "
            "to your Streamlit secrets or environment variables before using the app."
        )
        st.stop()

    if not os.path.exists(DB_PATH):
        st.error(
            f"Could not find the credit risk database at `{DB_PATH}`. "
            "Place `credit_risk_portfolio.db` alongside `app.py`, or set the `DB_PATH` secret/env var."
        )
        st.stop()

    llm, evaluator_llm = get_llms()
    conn = get_connection(DB_PATH)
    verified_query_library = build_verified_query_library()

    # ---- Sidebar: library and schema reference ----------------------------------------
    with st.sidebar:
        st.header("About")
        st.write(
            "This PoC covers the commercial lending portfolio only, is strictly read-only, "
            "and does not replace human judgment for complex credit-risk decisions."
        )

        # --- BLOCK FOR EXAMPLE QUESTIONS ---
      st.subheader("Example Questions")
      example_queries = [
          (
              "How much of our book is in real estate, and how much of that is"
              " non-performing?"
          ),
          "Show me how our overdue book is distributed across DPD buckets.",
          (
              "List every restructured loan, flag which are impaired, and show"
              " the overall impaired percentage."
          ),
          (
              "Show me the average interest rate by sector, sorted from highest"
              " to lowest."
          ),
          (
              "How has the Stage 3 book size and its expected credit loss moved"
              " over the last four quarters?"
          ),
      ]

      selected_example = st.selectbox(
          "Select a sample question:",
          ["-- Choose an example --"] + example_queries,
          key="example_selector",
      )

      if selected_example != "-- Choose an example --":
        st.session_state["user_input_query"] = selected_example
      # ---------------------------------------------
          
        st.subheader("Verified Query Library")
        for qid, entry in verified_query_library.items():
            with st.expander(qid):
                st.write(entry["description"])
                st.code(entry["sql"].strip(), language="sql")

        with st.expander("Database schema used by the LLM"):
            st.code(DATABASE_SCHEMA, language="text")

    # ---- Tabs ---------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["💬 Ask a Question", "✅ Test Cases & Evaluation", "🧾 Audit Trail"])

    with tab1:
        render_ask_tab(conn, verified_query_library, llm, evaluator_llm)

    with tab2:
        render_evaluation_tab(conn, verified_query_library, llm, evaluator_llm)

    with tab3:
        render_audit_tab()


if __name__ == "__main__":
    main()
