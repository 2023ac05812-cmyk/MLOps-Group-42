# api/app.py
import os, time, sqlite3, logging
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import joblib
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse

# logging
logging.basicConfig(level=logging.INFO, filename="api.log", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# metrics
REQUESTS = Counter("prediction_requests_total", "Total prediction requests")
LATENCY = Histogram("prediction_latency_seconds", "Request latency seconds")

app = FastAPI()
DB = "requests.db"
MODEL_URI = os.getenv("MODEL_URI")  # e.g. "models:/IrisModel/1" or runs:/<run>/model
_model = None

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, input TEXT, output TEXT, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()

def get_model():
    global _model
    if _model is not None:
        return _model
    # try MLflow model registry/URI
    if MODEL_URI:
        try:
            _model = mlflow.pyfunc.load_model(MODEL_URI)
            logger.info(f"Loaded model from {MODEL_URI}")
            return _model
        except Exception as e:
            logger.warning(f"Failed to load from MLflow URI {MODEL_URI}: {e}")
    # fallback local
    local_pkl = "models/best_model.pkl"
    if os.path.exists(local_pkl):
        _model = joblib.load(local_pkl)
        logger.info("Loaded local model")
        return _model
    raise RuntimeError("No model found. Train and provide MODEL_URI or models/best_model.pkl")

class InputData(BaseModel):
    sepallength: float
    sepalwidth: float
    petallength: float
    petalwidth: float

@app.on_event("startup")
def startup():
    init_db()
    get_model()

@app.post("/predict")
def predict(payload: InputData):
    REQUESTS.inc()
    t0 = time.time()
    m = get_model()
    x = [[payload.sepallength, payload.sepalwidth, payload.petallength, payload.petalwidth]]
    # If mlflow pyfunc: use .predict; sklearn models accept same
    try:
        y = m.predict(x)
    except Exception:
        y = m.predict_proba(x) if hasattr(m, "predict_proba") else ["error"]
    latency = time.time() - t0
    LATENCY.observe(latency)
    out = str(y.tolist() if hasattr(y, "tolist") else list(y))
    logger.info(f"input={x} output={out}")
    conn = sqlite3.connect(DB)
    conn.execute("INSERT INTO requests (input, output) VALUES (?, ?)", (str(x), out))
    conn.commit()
    conn.close()
    return {"prediction": out, "latency": latency}

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
