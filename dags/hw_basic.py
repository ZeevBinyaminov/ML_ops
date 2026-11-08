from datetime import timedelta

import pendulum
import requests
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import BaseHook, Param, dag, task

import batch_common as core

HTTP_TIMEOUT = 10
MODEL_VERSION = "v1"

default_args = {
    "retries": 2,
    "retry_delay": timedelta(seconds=5),
    "execution_timeout": timedelta(minutes=2),
}


def _get(url):
    # 404/503 должны ронять таск, а не давать пустой день
    response = requests.get(url, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response


@dag(
    dag_id="hw_basic_batch_inference",
    start_date=pendulum.datetime(2026, 9, 1, tz="Europe/Moscow"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["homework", "basic"],
    default_args=default_args,
    params={
        "business_date": Param("2026-09-01", type="string", format="date"),
        "batch_size": Param(32, type="integer", minimum=1, maximum=1000),
    },
)
def hw_basic_batch_inference():

    @task
    def extract(**context):
        day = core.business_date(context["params"]["business_date"],
                                 context.get("data_interval_start"))
        host = BaseHook.get_connection("source_api").host.rstrip("/")

        payload = _get(f"{host}/basic/{day}.json").json()
        if payload.get("business_date") != day:
            raise ValueError(f"Source returned {payload.get('business_date')!r}, expected {day!r}")

        model_response = _get(f"{host}/models/{MODEL_VERSION}.json")
        return core.save({
            "business_date": day,
            "rows": payload["rows"],
            "model": model_response.json(),
            # хэшируем сырые байты ответа, не json.dumps
            "model_sha256": core.digest(model_response.content),
        })

    @task(retries=0)  # кривые данные ретраем не починить
    def infer(ref, **context):
        data = core.read(ref)
        rows = core.validate_rows(data["rows"], data["business_date"])
        predictions = core.infer_batches(rows, data["model"], context["params"]["batch_size"])
        core.validate_predictions(predictions, [row["customer_id"] for row in rows])
        return core.save({
            "business_date": data["business_date"],
            "model_version": MODEL_VERSION,
            "predictions": predictions,
            "model_sha256": data["model_sha256"],
        })

    @task
    def publish(ref):
        result = core.read(ref)
        conn = PostgresHook(postgres_conn_id="warehouse").get_conn()
        try:
            # до replace_day на conn ничего не выполняем, иначе будет savepoint
            core.replace_day(conn, result["business_date"], result["model_version"],
                             result["predictions"])
        finally:
            conn.close()
        return {
            "business_date": result["business_date"],
            "model_version": result["model_version"],
            "rows": len(result["predictions"]),
        }

    publish(infer(extract()))


hw_basic_batch_inference()
