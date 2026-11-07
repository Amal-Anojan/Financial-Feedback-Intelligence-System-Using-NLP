# Database setup: MySQL, Workbench, WAMP or XAMPP

## What each component does

MySQL/MariaDB Server stores records. Workbench, mysql CLI and phpMyAdmin are administration clients. PyMySQL lets Python connect. Streamlit serves the website itself; Apache/PHP are not needed to serve app.py. XAMPP's phpMyAdmin may need Apache, independently of Streamlit.

1. Start your database service. If MySQL standalone and XAMPP/WAMP are both installed, confirm which server is listening and its port. Do not assume both use 3306.
2. Connect as an administrator in Workbench, mysql CLI or phpMyAdmin SQL tab.
3. Run the following after replacing the password placeholder with your own password. Do not commit real passwords.

```sql
CREATE DATABASE IF NOT EXISTS finsentiment
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'finsentiment_app'@'localhost'
  IDENTIFIED BY 'REPLACE_WITH_A_STRONG_LOCAL_PASSWORD';
GRANT SELECT, INSERT, UPDATE, CREATE, REFERENCES
  ON finsentiment.* TO 'finsentiment_app'@'localhost';
```

IF NOT EXISTS does not change an existing user's password. Use the account's actual password, or deliberately reset it as administrator. Do not use a blank-password root account for the app. Remote app/database hosts require a correctly scoped account host and protected network configuration; this recipe is for a local server.

4. Copy `database.example.toml` to `database.toml`; set host, port, database name, user and password to match that server. Do not put passwords inside Python code. XAMPP/WAMP does not require different Python code; configure its server account/port. MariaDB compatibility should be smoke-tested against the actual installed version; this project has not established universal compatibility.
5. Install requirements in the same Python environment as Streamlit. Start the app. SQLAlchemy creates three tables automatically: `reviews`, `review_analysis`, `review_analysis_evidence`. You create the database/user first; the app creates tables. There is no need to manually import CSV in Workbench first.
6. Import CSV through the dashboard and click analysis. A selected file is not an imported dataset until you click import.

Without database.toml, SQLite is used for local testing. Changing backend does not migrate old records automatically.

## Verify and query

```sql
USE finsentiment;
SHOW TABLES;
SELECT COUNT(*) FROM reviews;
SELECT a.status, COUNT(*) FROM review_analysis a GROUP BY a.status;
SELECT r.id, r.review_text, a.risk_score, a.intent, a.sentiment,
       a.root_cause, a.model_version, a.processed_at, e.details_json
FROM reviews r JOIN review_analysis a ON a.review_id = r.id
LEFT JOIN review_analysis_evidence e ON e.review_id = r.id
WHERE a.status = 'completed' AND a.risk_score > 7
ORDER BY a.risk_score DESC;
```

1146: tables were not initialized in this database, or wrong database selected. 1142: verify grants and which user/host is authenticated. 1045: credentials/account host mismatch. Connection refused: server/service/port. Missing pymysql: run `python -m pip install -r requirements.txt` in the app environment.

Schema creation is not a general migration engine. Back up company data before future schema changes. Unfinished claims are eligible after a 10-minute lease; retries preserve previously committed results. Evidence is saved transactionally with final analysis. SQL tests use SQLite fixtures; real MySQL outage/restart behavior still needs environment-specific integration testing.

Sources: [MySQL CREATE USER](https://dev.mysql.com/doc/refman/8.4/en/create-user.html), [MySQL GRANT](https://dev.mysql.com/doc/refman/8.4/en/grant.html), [XAMPP Windows FAQ](https://www.apachefriends.org/faq_windows.html).
