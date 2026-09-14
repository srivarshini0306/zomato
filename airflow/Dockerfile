# Airflow 3.x image + Snowflake provider + FAB auth + Groq + dbt
FROM apache/airflow:3.0.3


# ------------------------------------------------------------
# Providers + Groq
# ------------------------------------------------------------

RUN pip install --no-cache-dir \
      apache-airflow-providers-snowflake \
      apache-airflow-providers-fab \
      groq


# ------------------------------------------------------------
# dbt
#
# dbt is installed in its own virtual environment so that
# dbt dependencies don't conflict with Airflow dependencies.
#
# DAGs can call:
# /opt/airflow/dbt_venv/bin/dbt
# ------------------------------------------------------------

RUN python -m venv /opt/airflow/dbt_venv && \
    /opt/airflow/dbt_venv/bin/pip install --no-cache-dir "dbt-snowflake==1.8.*"