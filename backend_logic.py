# 📁 backend_logic.py (LangGraph – Enhanced Output for UI Display)
import os
import re
import json
import uuid
import base64
import boto3
from datetime import datetime, date
from typing import TypedDict, Annotated
import operator

from snowflake.snowpark import Session, WhenMatchedClause, WhenNotMatchedClause
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# Configuration: Set these in your .env file (do not commit .env to version control)
# Example .env entries:

TABLE_NAME = os.getenv("DRIVERS_LICENSE_TABLE")
if not TABLE_NAME:
    raise RuntimeError("DRIVERS_LICENSE_TABLE environment variable is not set. Please set it in your .env file.")
STAGE_NAME = os.getenv("SNOWFLAKE_STAGE")
if not STAGE_NAME:
    raise RuntimeError("SNOWFLAKE_STAGE environment variable is not set. Please set it in your .env file.")
LOCAL_PATH = os.getenv("LOCAL_TMP_PATH")
if not LOCAL_PATH:
    raise RuntimeError("LOCAL_TMP_PATH environment variable is not set. Please set it in your .env file.")

# Load Snowflake credentials
conn = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}
session = Session.builder.configs(conn).create()

# Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
MODEL_ID = "amazon.nova-lite-v1:0"

# Shared types
class State(TypedDict):
    dl_path: str
    claim_path: str
    car_path: str
    car_local: str
    dl: dict
    claim: dict
    car: dict
    steps: Annotated[list, operator.add]
    files: dict
    comparison: dict
    decision: str
    email: str

# --------------------- Helpers ---------------------
def safe_extract(field):
    """Extract value from Document AI output."""
    if not field:
        return None
    if isinstance(field, list):
        for item in field:
            if isinstance(item, dict) and "value" in item:
                return item["value"]
        return field[0] if field else None
    if isinstance(field, dict):
        return field.get("value")
    return field

def parse_date(value):
    """Parse dates in multiple formats and normalize to YYYY-MM-DD."""
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    formats = ["%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"]
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            if "%y" in fmt and dt.year < 1930:  # Fix 2-digit years
                dt = dt.replace(year=dt.year + 100)
            return dt.date()
        except ValueError:
            continue
    print(f"Could not parse date: {value}")
    return None

def normalize_dates(record: dict):
    """Convert all date objects to ISO strings."""
    for key, value in record.items():
        if isinstance(value, (datetime, date)):
            record[key] = value.isoformat()
    return record

def safe_json_dumps(data):
    """Dump JSON with date handling."""
    def default(o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return str(o)
    return json.dumps(data, default=default)

def upload_to_stage(file, filename):
    local_path = os.path.join(LOCAL_PATH, filename)
    with open(local_path, "wb") as f:
        f.write(file.read())
    session.file.put(local_path, STAGE_NAME, overwrite=True, auto_compress=False)
    return filename, local_path

def normalize_comparison(data):
    """Ensure the comparison JSON has all expected fields with consistent types."""
    normalized = {
        "name": {"claim_table": False},
        "license_no": {"claim_table": False, "dl_table": False},
        "address": {"claim_table": False, "dl_table": False},
        "car_make": {"claim_car": False},
        "car_model": {"claim_car": False},
        "car_color": {"claim_car": False},
        "damage_details": {"claim_car": False},
        "license_validity": False
    }
    if not isinstance(data, dict):
        return normalized
    for key in normalized:
        if key == "license_validity":
            normalized[key] = bool(data.get(key, False))
        else:
            field_data = data.get(key, {})
            if not isinstance(field_data, dict):
                field_data = {}
            for subkey in normalized[key]:
                normalized[key][subkey] = bool(field_data.get(subkey, False))
    return normalized

# --------------------- Nodes ---------------------
def upload_all(state: State):
    files = state.get("files")
    if not files:
        raise ValueError("Missing 'files' in workflow state.")
    dl_name, _ = upload_to_stage(files["dl"], f"{uuid.uuid4()}_{files['dl'].name}")
    claim_name, _ = upload_to_stage(files["claim"], f"{uuid.uuid4()}_{files['claim'].name}")
    car_name, car_path = upload_to_stage(files["car"], f"{uuid.uuid4()}_{files['car'].name}")
    return {
        "dl_path": dl_name,
        "claim_path": claim_name,
        "car_path": car_name,
        "car_local": car_path,
        "steps": ["Uploaded"],
    }

def extract_dl(state: State):
    path = state["dl_path"]
    try:
        sql = f"SELECT LICENSE_DATA!PREDICT(GET_PRESIGNED_URL({STAGE_NAME}, '{path}'), 1) AS result"
        result = json.loads(session.sql(sql).collect()[0]["RESULT"])
    except Exception as e:
        raise RuntimeError(f"Failed to run LICENSE_DATA PREDICT: {e}")

    print("\nRaw DL Data:", json.dumps(result, indent=2))
    raw_dob = safe_extract(result.get("dob"))
    raw_issue = safe_extract(result.get("issue_date"))
    raw_expiry = safe_extract(result.get("expiry_date"))

    dl_record_full = {
        "FULL_NAME": safe_extract(result.get("name")) or "Unknown",
        "LICENSE_NUMBER": safe_extract(result.get("license_no")),
        "ADDRESS": safe_extract(result.get("address")),
        "DATE_OF_BIRTH": parse_date(raw_dob),
        "ISSUE_DATE": parse_date(raw_issue),
        "EXPIRY_DATE": parse_date(raw_expiry),
        "ENDORSEMENTS": safe_extract(result.get("endorsements")) or "",
        "SEX": safe_extract(result.get("sex")),
        "HEIGHT": safe_extract(result.get("height")),
    }
    dl_record_full = normalize_dates(dl_record_full)

    # Fetch table row for this license number
    table_row = None
    license_number = dl_record_full["LICENSE_NUMBER"]
    if license_number:
        import re
        if not re.match(r'^[\w-]+$', license_number):
            raise ValueError("Invalid license number format")
        try:
            table_result = session.table(TABLE_NAME).filter(
                f"LICENSE_NUMBER = '{license_number}'"
            ).collect()
            if table_result:
                table_row = table_result[0].asDict()
                table_row = normalize_dates(table_row)
        except Exception as e:
            print(f"Could not fetch table row for LICENSE_NUMBER {license_number}: {e}")

    if not dl_record_full["LICENSE_NUMBER"]:
        print("Skipping merge: No LICENSE_NUMBER found.")
        return {"dl": dl_record_full, "table_row": table_row, "steps": ["DL Extracted (Not Stored: Missing License Number)"]}

    try:
        df_new = session.create_dataframe([dl_record_full])
        target = session.table(TABLE_NAME)
        target.merge(
            df_new,
            target["LICENSE_NUMBER"] == df_new["LICENSE_NUMBER"],
            [WhenMatchedClause().update(dl_record_full),
             WhenNotMatchedClause().insert(dl_record_full)],
        )
    except Exception as e:
        print(f"Failed to merge DL record: {e}")

    print("\nParsed & Stored DL Data:", json.dumps(dl_record_full, indent=2))
    return {"dl": dl_record_full, "table_row": table_row, "steps": ["Driver's License Extracted & Stored"]}

def extract_claim(state: State):
    path = state["claim_path"]
    sql = f"SELECT CLAIMS_DATA!PREDICT(GET_PRESIGNED_URL({STAGE_NAME}, '{path}'), 1) AS result"
    result = json.loads(session.sql(sql).collect()[0]["RESULT"])
    for field in ["description", "vehicle"]:
        text = result.get(field, [{}])[0].get("value", "")
        match = re.search(r"Color:\s*(.+?)(,|$)", text, re.IGNORECASE)
        if match:
            result["car_color"] = [{"value": match.group(1).strip()}]
            break
    print("\nExtracted Claim Data:", json.dumps(result, indent=2))
    return {"claim": result, "steps": ["Claim Extracted"]}

def extract_car(state: State):
    def get_image_mime(ext: str):
        mapping = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
        return mapping.get(ext.lower())
    path = state["car_local"]
    ext = path.split(".")[-1]
    mime_type = get_image_mime(ext)
    if not mime_type:
        raise ValueError(f"Unsupported image format: {ext}")
    with open(path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "schemaVersion": "messages-v1",
        "inferenceConfig": {"max_new_tokens": 500, "temperature": 0.3},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"image": {"format": ext, "source": {"bytes": b64_img}}},
                    {"text": "Analyze the car image and return car type, make, model, visible damage, severity, and color in JSON."},
                ],
            }
        ],
    }
    response = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(payload))
    car_result = json.loads(response["body"].read())
    raw_text = car_result.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        car_data = json.loads(match.group(0)) if match else {"raw_output": raw_text}
    except json.JSONDecodeError:
        car_data = {"raw_output": raw_text}
    print("\nExtracted Car Analysis:", json.dumps(car_data, indent=2))
    return {"car": car_data, "steps": ["Car Analyzed"]}

def compare_and_email(state: State):
    dl = state["dl"]
    claim = state["claim"]
    car = state["car"]
    table_row = state.get("table_row")
    today_str = datetime.today().date().isoformat()

    prompt = f"""
Compare the following fields and return a JSON with match results for each:

- name: claim document vs table
- license_no: claim document vs table
- license_no: driving license vs table
- address: claim document vs table
- address: driving license vs table
- car_make: claim document vs car image
- car_model: claim document vs car image
- car_color: claim document vs car image
- damage_details: claim document vs car image
- license_validity: Is the expiry_date in the table after today's date ({today_str})?

Return JSON only, like:
{{
  "name": {{"claim_table": true/false}},
  "license_no": {{"claim_table": true/false, "dl_table": true/false}},
  "address": {{"claim_table": true/false, "dl_table": true/false}},
  "car_make": {{"claim_car": true/false}},
  "car_model": {{"claim_car": true/false}},
  "car_color": {{"claim_car": true/false}},
  "damage_details": {{"claim_car": true/false}},
  "license_validity": true/false
}}

DL: {safe_json_dumps(dl)}
CLAIM: {safe_json_dumps(claim)}
CAR: {safe_json_dumps(car)}
TABLE: {safe_json_dumps(table_row) if table_row else '{}'}
"""
    comparison_payload = {
        "schemaVersion": "messages-v1",
        "inferenceConfig": {"max_new_tokens": 1200, "temperature": 0.2},
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
    }
    response = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(comparison_payload))
    raw_text = json.loads(response["body"].read()).get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        comparison = json.loads(match.group(0)) if match else {}
    except:
        comparison = {"raw_output": raw_text}

    # Normalize comparison
    comparison = normalize_comparison(comparison)

    # Decision
       # Only accept if all required fields match
    all_match = True
    required_fields = [
        ("name", ["claim_table"]),
        ("license_no", ["claim_table", "dl_table"]),
        ("address", ["claim_table", "dl_table"]),
        ("car_make", ["claim_car"]),
        ("car_model", ["claim_car"]),
        ("car_color", ["claim_car"]),
        ("damage_details", ["claim_car"]),
        ("license_validity", []),
    ]
    for field, subfields in required_fields:
        field_result = comparison.get(field, {})
        if field == "license_validity":
            # Coerce any type to boolean silently
            if isinstance(field_result, dict):
                field_result = field_result.get("value", False)
            all_match = bool(field_result)
        else:
            for sub in subfields:
                if field_result.get(sub) is not True:
                    all_match = False

    decision = "Claim Accepted" if all_match else "Claim Rejected"

    # Email
    email_prompt = f"""
Write a short professional email summarizing the claim review.

Comparison:
{safe_json_dumps(comparison)}

Decision: {decision}
"""
    email_payload = {
        "schemaVersion": "messages-v1",
        "inferenceConfig": {"max_new_tokens": 500, "temperature": 0.3},
        "messages": [{"role": "user", "content": [{"text": email_prompt}]}],
    }
    email_response = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(email_payload))
    email_text = json.loads(email_response["body"].read()).get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")

    return {
        "comparison": comparison,
        "decision": decision,
        "email": email_text,
        "steps": ["Compared & Emailed"],
    }

# --------------------- Graph ---------------------
def run_claim_processing_workflow(files):
    class WorkflowState(State):
        files: dict
    builder = StateGraph(WorkflowState)
    builder.add_node("upload", upload_all)
    builder.add_node("extract_dl", extract_dl)
    builder.add_node("extract_claim", extract_claim)
    builder.add_node("extract_car", extract_car)
    builder.add_node("compare_email", compare_and_email)
    builder.set_entry_point("upload")
    builder.add_edge("upload", "extract_dl")
    builder.add_edge("extract_dl", "extract_claim")
    builder.add_edge("extract_claim", "extract_car")
    builder.add_edge("extract_car", "compare_email")
    builder.add_edge("compare_email", END)
    graph = builder.compile()
    return graph.invoke({"files": files, "steps": []})
