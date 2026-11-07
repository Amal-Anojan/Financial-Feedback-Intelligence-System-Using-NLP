"""Small SQLAlchemy store shared by the dashboard and one-shot processing command."""
import hashlib
import json
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and earlier
    import tomli as tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, MetaData, String,
                        Table, Text, URL, and_, create_engine, event, func, or_, select)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.exc import DBAPIError, IntegrityError

from review_analyzer import ROOT

metadata = MetaData()
large_text = Text().with_variant(LONGTEXT(), "mysql")
reviews = Table("reviews", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_key", String(64), unique=True, nullable=False),
    Column("source", String(200), nullable=False),
    Column("external_id", String(255)),
    Column("review_text", large_text, nullable=False),
    Column("raw_row_json", large_text, nullable=False),
    Column("received_at", DateTime, nullable=False, index=True),
    Column("imported_at", DateTime, nullable=False))
analyses = Table("review_analysis", metadata,
    Column("review_id", Integer, ForeignKey("reviews.id"), primary_key=True),
    Column("status", String(20), nullable=False, index=True),
    Column("attempts", Integer, nullable=False, default=0),
    Column("claim_token", String(36)), Column("lease_until", DateTime),
    Column("risk_score", Float), Column("intent", String(100)),
    Column("sentiment", String(100)), Column("root_cause", String(100)),
    Column("intent_confidence", Float), Column("sentiment_confidence", Float),
    Column("root_cause_confidence", Float),
    Column("model_version", String(100)), Column("processing_ms", Float),
    Column("processed_at", DateTime), Column("error", Text),
    Column("suggested_action", Text), Column("action_owner", String(200)),
    Column("suggestion_version", String(100)))

# Separate table avoids ALTER permissions/migration of existing analysis columns.
analysis_evidence = Table("review_analysis_evidence", metadata,
    Column("review_id", Integer, ForeignKey("review_analysis.review_id"), primary_key=True),
    Column("details_json", large_text, nullable=False))


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def database_error_message(exc):
    """Explain common setup failures without exposing SQL, parameters or credentials."""
    original = getattr(exc, "orig", exc)
    args = getattr(original, "args", ())
    code = args[0] if args and isinstance(args[0], int) else None
    messages = {
        1045: "MySQL rejected the login (1045). The user and password in database.toml must match the MySQL account. If CREATE USER used CHANGE_THIS_PASSWORD literally, change that account password in Workbench and put the same password in database.toml. Restart Streamlit afterward.",
        1044: "MySQL denied access to the database (1044). Grant the application account access to finsentiment using MYSQL_SETUP.md.",
        1049: "The configured MySQL database does not exist (1049). Create finsentiment and check the name in database.toml.",
        1142: "MySQL denied a table operation (1142). The application account needs SELECT, INSERT, UPDATE, CREATE and REFERENCES permissions on finsentiment.*.",
        1146: "A required table is missing (1146). Restart Streamlit after fixing the database connection and permissions; the dashboard creates reviews and review_analysis on startup. Run the result queries only after those tables exist.",
        2003: "Cannot reach MySQL Server (2003). Start the server and check host and port in database.toml.",
        2006: "The MySQL connection was lost (2006). Check the server and try again. Previously saved results remain stored.",
        2013: "The MySQL connection was interrupted (2013). Check the server and try again. Previously saved results remain stored.",
    }
    if code in messages:
        return messages[code]
    if isinstance(exc, ModuleNotFoundError) and exc.name == "pymysql":
        return "The MySQL driver is missing. Run python -m pip install -r requirements.txt in the same environment that runs Streamlit, then restart it."
    return f"Database or model operation could not finish ({type(exc).__name__}). Check database.toml, the database server, and model files. Previously saved results are retained. Retry after recovery; unfinished claims become eligible after 10 minutes."


def transient_database_error(exc):
    if not isinstance(exc, DBAPIError):
        return False
    code = exc.orig.args[0] if exc.orig and exc.orig.args else None
    return exc.connection_invalidated or code in {1205, 1213, 2002, 2003, 2006, 2013} or "database is locked" in str(exc.orig).lower()


def database_retry(operation, sleep=time.sleep):
    for attempt in range(3):
        try:
            return operation()
        except DBAPIError as exc:
            if attempt == 2 or not transient_database_error(exc):
                raise
            sleep(0.25 * (2 ** attempt))


def configured_engine(config_path=ROOT / "database.toml"):
    config = {}
    if Path(config_path).exists():
        with open(config_path, "rb") as handle:
            config = tomllib.load(handle).get("database", {})
    backend = config.get("backend", "sqlite")
    if backend == "mysql":
        required = ("host", "user", "password", "name")
        if any(not config.get(key) for key in required):
            raise ValueError("Complete the MySQL settings in database.toml; see database.example.toml.")
        url = URL.create("mysql+pymysql", username=config["user"], password=config["password"],
                         host=config["host"], port=int(config.get("port", 3306)),
                         database=config["name"], query={"charset": "utf8mb4"})
        return create_engine(url, pool_pre_ping=True, hide_parameters=True,
                             connect_args={"connect_timeout": 5, "read_timeout": 15, "write_timeout": 15})
    if backend != "sqlite":
        raise ValueError("Database backend must be mysql or sqlite.")
    folder = ROOT / "runtime"
    folder.mkdir(exist_ok=True)
    return sqlite_engine(folder / "reviews.sqlite3")


def sqlite_engine(path):
    engine = create_engine(URL.create("sqlite", database=str(path)), hide_parameters=True,
                           connect_args={"timeout": 15})
    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")
    return engine


class ReviewStore:
    def __init__(self, engine):
        self.engine = engine

    def initialize(self):
        database_retry(lambda: metadata.create_all(self.engine))

    def import_rows(self, rows, source, text_column, id_column=None, date_column=None):
        """One atomic import; existing source IDs are never overwritten."""
        if not source.strip() or len(source) > 200:
            raise ValueError("Choose a stable source name of 1–200 characters.")
        prepared = []
        for row in rows:
            if text_column not in row:
                raise ValueError(f"CSV has no {text_column!r} column.")
            raw = json.dumps(row, ensure_ascii=False, sort_keys=True)
            external_id = str(row.get(id_column, "")) if id_column else None
            if id_column and (not external_id.strip() or len(external_id) > 255):
                raise ValueError("Every selected review ID must be nonempty and at most 255 characters.")
            identity = [source, "id", external_id] if id_column else [source, "row", raw]
            key = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()
            received = utcnow()
            if date_column and row.get(date_column):
                try:
                    value = datetime.fromisoformat(str(row[date_column]).replace("Z", "+00:00"))
                    received = value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
                except ValueError:
                    raise ValueError("Review dates must use ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS). Leave the date column unselected to use import time.") from None
            prepared.append(dict(source_key=key, source=source, external_id=external_id,
                                 review_text=str(row.get(text_column) or ""), raw_row_json=raw,
                                 received_at=received, imported_at=utcnow()))
        def save():
            added = skipped = conflicts = 0
            with self.engine.begin() as connection:
                for values in prepared:
                    existing = connection.execute(select(reviews.c.raw_row_json).where(
                        reviews.c.source_key == values["source_key"])).first()
                    if existing:
                        skipped += 1
                        conflicts += existing[0] != values["raw_row_json"]
                        continue
                    try:
                        with connection.begin_nested():
                            connection.execute(reviews.insert().values(**values))
                        added += 1
                    except IntegrityError:
                        # A concurrent import may have inserted this source ID.
                        existing = connection.execute(select(reviews.c.raw_row_json).where(
                            reviews.c.source_key == values["source_key"])).first()
                        if not existing:
                            raise
                        skipped += 1
                        conflicts += existing[0] != values["raw_row_json"]
            return {"imported": added, "duplicates_skipped": skipped, "changed_rows_preserved": conflicts}
        return database_retry(save)

    def max_review_id(self):
        def query():
            with self.engine.connect() as connection:
                return connection.scalar(select(func.max(reviews.c.id))) or 0
        return database_retry(query)

    def candidates(self, after, maximum, retry_failed=False):
        eligible = or_(analyses.c.review_id.is_(None),
                       and_(analyses.c.status == "processing", analyses.c.lease_until < utcnow()))
        if retry_failed:
            eligible = or_(eligible, analyses.c.status == "failed")
        statement = select(reviews).outerjoin(analyses).where(
            reviews.c.id > after, reviews.c.id <= maximum, eligible).order_by(reviews.c.id).limit(100)
        def query():
            with self.engine.connect() as connection:
                return list(connection.execute(statement).mappings())
        return database_retry(query)

    def claim(self, review_id, retry_failed=False):
        token = str(uuid4())
        def acquire():
            now = utcnow()
            with self.engine.begin() as connection:
                current = connection.execute(select(analyses).where(analyses.c.review_id == review_id)).mappings().first()
                # Handles a lost acknowledgement of our own committed claim.
                if current and current["claim_token"] == token:
                    return token
                values = dict(status="processing", claim_token=token,
                              lease_until=now + timedelta(minutes=10), error=None)
                if not current:
                    try:
                        with connection.begin_nested():
                            connection.execute(analyses.insert().values(review_id=review_id, attempts=1, **values))
                        return token
                    except IntegrityError:
                        return None
                eligible = and_(analyses.c.status == "processing", analyses.c.lease_until < now)
                if retry_failed:
                    eligible = or_(eligible, analyses.c.status == "failed")
                updated = connection.execute(analyses.update().where(
                    analyses.c.review_id == review_id, eligible).values(
                        attempts=analyses.c.attempts + 1, **values))
                return token if updated.rowcount == 1 else None
        return database_retry(acquire)

    def finish(self, review_id, token, values):
        values = dict(values)
        evidence = values.pop("evidence", None)
        def save():
            with self.engine.begin() as connection:
                updated = connection.execute(analyses.update().where(
                    analyses.c.review_id == review_id, analyses.c.claim_token == token).values(
                        **values, lease_until=None, processed_at=utcnow()))
                if updated.rowcount == 1 and evidence is not None:
                    encoded = json.dumps(evidence, ensure_ascii=False, allow_nan=False)
                    exists = connection.execute(select(analysis_evidence.c.review_id).where(
                        analysis_evidence.c.review_id == review_id)).first()
                    if exists:
                        connection.execute(analysis_evidence.update().where(
                            analysis_evidence.c.review_id == review_id).values(details_json=encoded))
                    else:
                        connection.execute(analysis_evidence.insert().values(review_id=review_id, details_json=encoded))
                return updated.rowcount == 1
        return database_retry(save)

    def dashboard(self, filters=None, limit=1000):
        filters = filters or {}
        status = func.coalesce(analyses.c.status, "pending")
        statement = select(reviews.c.id, reviews.c.source, reviews.c.external_id,
            reviews.c.review_text, reviews.c.received_at, status.label("status"),
            *[c for c in analyses.c if c.name not in {"review_id", "status", "claim_token", "lease_until"}]
        ).outerjoin(analyses)
        for field in ("intent", "sentiment", "status"):
            if filters.get(field):
                statement = statement.where((status if field == "status" else analyses.c[field]) == filters[field])
        if filters.get("risk") == "Above 7":
            statement = statement.where(analyses.c.risk_score > 7)
        elif filters.get("risk") == "4–7":
            statement = statement.where(analyses.c.risk_score.between(4, 7))
        elif filters.get("risk") == "Below 4":
            statement = statement.where(analyses.c.risk_score < 4)
        if filters.get("start"):
            statement = statement.where(reviews.c.received_at >= filters["start"])
        if filters.get("end"):
            statement = statement.where(reviews.c.received_at < filters["end"])
        def query():
            with self.engine.connect() as connection:
                count = connection.scalar(select(func.count()).select_from(statement.subquery()))
                rows = list(connection.execute(statement.order_by(reviews.c.id.desc()).limit(limit)).mappings())
                totals = dict(connection.execute(select(status, func.count()).select_from(
                    reviews.outerjoin(analyses)).group_by(status)).all())
                options = {field: list(connection.scalars(select(analyses.c[field]).where(
                    analyses.c[field].is_not(None)).distinct().order_by(analyses.c[field]))) for field in ("sentiment", "intent")}
                return rows, count, totals, options
        return database_retry(query)
