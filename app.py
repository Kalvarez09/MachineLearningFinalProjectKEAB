import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_football_classifier.keras"
MAPPING_PATH = BASE_DIR / "class_mapping.json"
CONFIG_PATH = BASE_DIR / "deployment_config.json"

DISPLAY_NAMES = {
    "Corner_Kick": "Corner Kick",
    "Free_Kick": "Free Kick",
    "Penalty_kick": "Penalty Kick",
    "Red_Card": "Red Card",
    "Yellow_Card": "Yellow Card",
    "substitute": "Substitution",
    "tackle": "Tackle",
}

with MAPPING_PATH.open("r", encoding="utf-8") as file:
    mapping = json.load(file)

CLASS_NAMES = [
    name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))
]

app = FastAPI(title="Football Event Intelligence", version="1.0.0")
model: Any = None
model_error: str | None = None


class PredictionItem(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    top_predictions: list[PredictionItem]
    model: str
    demo: bool = False


@app.on_event("startup")
def startup() -> None:
    global model, model_error
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow import keras
        model = keras.models.load_model(MODEL_PATH, compile=False)
        model_error = None
    except Exception as exc:
        model = None
        model_error = str(exc)


def prepare_image(raw: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB")
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "Invalid image file.") from exc
    return np.expand_dims(array, axis=0)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) != len(CLASS_NAMES):
        raise HTTPException(500, "Model output does not match class mapping.")
    if (
        np.any(values < 0)
        or np.any(values > 1)
        or not np.isclose(values.sum(), 1.0, atol=1e-3)
    ):
        values = np.exp(values - values.max())
        values = values / values.sum()
    return values


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Football Event Intelligence</title>
<style>
:root{--bg:#090b0a;--panel:#141815;--line:#ffffff18;--text:#f4f1e8;--muted:#a7afa8;--gold:#e8cb82;--green:#8fe0ad}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,sans-serif;color:var(--text);background:radial-gradient(circle at 15% 15%,#12322255,transparent 28%),radial-gradient(circle at 84% 10%,#6b4a1e44,transparent 30%),var(--bg);min-height:100vh}
nav{height:70px;border-bottom:1px solid var(--line);background:#080a09cc;backdrop-filter:blur(16px);display:flex;align-items:center;position:sticky;top:0;z-index:2}
.shell{width:min(1160px,calc(100% - 28px));margin:auto}.nav{display:flex;justify-content:space-between;align-items:center}.brand{font-weight:800}.links{display:flex;gap:22px;color:var(--muted);font-size:14px}
.hero{text-align:center;padding:82px 0 42px}.tag{display:inline-block;padding:8px 12px;border:1px solid #e8cb8240;border-radius:999px;color:var(--gold);font-size:12px;letter-spacing:.12em;text-transform:uppercase}.hero h1{font-size:clamp(44px,7vw,88px);line-height:.98;letter-spacing:-.05em;margin:24px 0 18px}.hero h1 span{color:var(--gold)}.hero p{max-width:720px;margin:auto;color:var(--muted);font-size:18px;line-height:1.7}
.stats{display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:30px}.stat{border:1px solid var(--line);border-radius:999px;padding:9px 13px;color:var(--muted);font-size:14px}.stat b{color:var(--text)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:22px 0 80px}.card{background:linear-gradient(180deg,#181c19f2,#0f1210f2);border:1px solid var(--line);border-radius:26px;box-shadow:0 25px 80px #0007;overflow:hidden}.head{padding:24px 24px 0}.head small{color:var(--gold);font-weight:800;letter-spacing:.12em;text-transform:uppercase}.head h2{font-size:28px;margin:8px 0}.head p{color:var(--muted);margin:0}
.drop{margin:22px;min-height:410px;border:1px dashed #e8cb8266;border-radius:22px;display:grid;place-items:center;text-align:center;padding:24px;background:#ffffff05;cursor:pointer}.drop.drag{border-color:var(--gold);background:#e8cb8210}.icon{width:64px;height:64px;border-radius:20px;display:grid;place-items:center;margin:0 auto 18px;background:#e8cb8210;border:1px solid #e8cb8244;color:var(--gold);font-size:28px}.drop p{color:var(--muted)}.browse{display:inline-block;padding:12px 18px;border-radius:999px;background:#e8cb8212;border:1px solid #e8cb8244;color:var(--gold);font-weight:800}input{display:none}.preview{width:100%;min-height:410px;object-fit:cover;border-radius:22px;display:none}
.actions{display:flex;gap:12px;padding:0 22px 22px}button{border:0;font:inherit;cursor:pointer}.primary{flex:1;min-height:54px;border-radius:16px;font-weight:900;background:linear-gradient(135deg,#f0d995,#b88635);color:#111}.primary:disabled{opacity:.45;cursor:not-allowed}.secondary{min-width:110px;border-radius:16px;background:#ffffff08;border:1px solid var(--line);color:var(--text)}
.error{display:none;margin:0 22px 22px;padding:13px 15px;border-radius:14px;color:#ff9d9d;background:#ff5b5b12;border:1px solid #ff5b5b33}.loading{display:none;padding:0 22px 22px;color:var(--gold)}
.empty{min-height:100%;display:grid;place-items:center;text-align:center;padding:40px}.empty .shield{width:66px;height:66px;border-radius:20px;display:grid;place-items:center;margin:auto auto 18px;color:var(--green);background:#8fe0ad10;border:1px solid #8fe0ad44;font-size:26px}.empty p{color:var(--muted);max-width:360px;line-height:1.6}
.result{display:none;padding:28px}.resulttop{display:flex;justify-content:space-between;gap:22px;padding-bottom:24px;border-bottom:1px solid var(--line)}.status{color:var(--green);font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:800}.label{color:var(--muted);margin-top:22px}.detected{font-size:56px;letter-spacing:-.05em;margin:6px 0}.model{color:var(--muted)}
.ring{--p:0deg;width:118px;height:118px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--gold) var(--p),#ffffff12 0);position:relative;flex:none}.ring:after{content:"";position:absolute;inset:8px;border-radius:50%;background:#111512}.ringin{z-index:1;text-align:center}.ringin b{display:block;font-size:27px}.ringin span{font-size:10px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}
.ranks{padding-top:24px}.ranks h4{color:var(--muted);font-size:12px;letter-spacing:.12em;text-transform:uppercase}.row{margin:17px 0}.rowtop{display:flex;justify-content:space-between;margin-bottom:8px}.score{color:var(--muted)}.bar{height:7px;border-radius:999px;background:#ffffff10;overflow:hidden}.bar span{display:block;height:100%;background:linear-gradient(90deg,#b88635,var(--gold));border-radius:999px}
footer{border-top:1px solid var(--line);padding:28px 0 40px;text-align:center;color:var(--muted);font-size:14px}
@media(max-width:900px){.grid{grid-template-columns:1fr}.links{display:none}.hero{padding-top:56px}}
</style>
</head>
<body>
<nav><div class="shell nav"><div class="brand">⚽ Football Event Intelligence</div><div class="links"><a href="#analyze">Analyze</a><a href="/docs">API Docs</a><a href="/health">Health</a></div></div></nav>
<main>
<section class="shell hero"><span class="tag">AI-powered football understanding</span><h1>See the event.<br><span>Understand the moment.</span></h1><p>Upload a football image and let the trained ConvNeXtTiny model classify the scene into one of seven football events.</p><div class="stats"><span class="stat"><b>7</b> classes</span><span class="stat"><b>84.6%</b> test accuracy</span><span class="stat"><b>ConvNeXtTiny</b></span></div></section>
<section class="shell grid" id="analyze">
<article class="card"><div class="head"><small>01 / Analyze</small><h2>Upload a football image</h2><p>JPG, PNG, or WEBP. Maximum 10 MB.</p></div>
<label class="drop" id="drop" for="file"><div id="prompt"><div class="icon">⇧</div><h3>Drag and drop your image</h3><p>or browse from your computer</p><span class="browse">Choose image</span></div><img class="preview" id="preview"></label>
<input id="file" type="file" accept="image/png,image/jpeg,image/webp">
<div class="actions"><button class="primary" id="predict" disabled>Run prediction</button><button class="secondary" id="reset">Reset</button></div><div class="loading" id="loading">Analyzing with the live model…</div><div class="error" id="error"></div></article>
<article class="card"><div class="empty" id="empty"><div><div class="shield">✓</div><h3>Your result will appear here</h3><p>The model returns the detected event, confidence score, and top three alternatives.</p></div></div>
<div class="result" id="result"><div class="resulttop"><div><div class="status">● Live model</div><div class="label">Detected event</div><h2 class="detected" id="detected">—</h2><div class="model" id="modelname">ConvNeXtTiny classification</div></div><div class="ring" id="ring"><div class="ringin"><b id="confidence">0%</b><span>Confidence</span></div></div></div><div class="ranks"><h4>Top predictions</h4><div id="ranking"></div></div></div></article>
</section>
</main>
<footer><div class="shell">Explainable Football Scene and Event Classification · Hugging Face Space</div></footer>
<script>
const file=document.getElementById("file"),drop=document.getElementById("drop"),preview=document.getElementById("preview"),prompt=document.getElementById("prompt"),predict=document.getElementById("predict"),reset=document.getElementById("reset"),loading=document.getElementById("loading"),error=document.getElementById("error"),empty=document.getElementById("empty"),result=document.getElementById("result"),detected=document.getElementById("detected"),confidence=document.getElementById("confidence"),ring=document.getElementById("ring"),ranking=document.getElementById("ranking"),modelname=document.getElementById("modelname");let selected=null;
function choose(f){error.style.display="none";if(!f)return;if(!f.type.startsWith("image/"))return fail("Choose a valid image.");if(f.size>10*1024*1024)return fail("Image must be smaller than 10 MB.");selected=f;preview.src=URL.createObjectURL(f);preview.style.display="block";prompt.style.display="none";predict.disabled=false}
function fail(t){error.textContent=t;error.style.display="block"}
function clearAll(){selected=null;file.value="";preview.src="";preview.style.display="none";prompt.style.display="block";predict.disabled=true;error.style.display="none";result.style.display="none";empty.style.display="grid"}
file.onchange=e=>choose(e.target.files[0]);["dragenter","dragover"].forEach(n=>drop.addEventListener(n,e=>{e.preventDefault();drop.classList.add("drag")}));["dragleave","drop"].forEach(n=>drop.addEventListener(n,e=>{e.preventDefault();drop.classList.remove("drag")}));drop.addEventListener("drop",e=>choose(e.dataTransfer.files[0]));reset.onclick=clearAll;
predict.onclick=async()=>{if(!selected)return;predict.disabled=true;loading.style.display="block";error.style.display="none";const form=new FormData();form.append("file",selected);try{const r=await fetch("/predict",{method:"POST",body:form});const d=await r.json();if(!r.ok)throw new Error(d.detail||"Prediction failed.");const pct=Math.round(d.confidence*100);detected.textContent=d.class_name;confidence.textContent=pct+"%";ring.style.setProperty("--p",(pct*3.6)+"deg");modelname.textContent=d.model+" classification";ranking.innerHTML=d.top_predictions.map((x,i)=>{const s=Math.round(x.confidence*100);return `<div class="row"><div class="rowtop"><b>0${i+1} · ${x.class_name}</b><span class="score">${s}%</span></div><div class="bar"><span style="width:${s}%"></span></div></div>`}).join("");empty.style.display="none";result.style.display="block"}catch(e){fail(e.message)}finally{loading.style.display="none";predict.disabled=false}};
</script>
</body>
</html>"""


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "model_unavailable",
        "model_loaded": model is not None,
        "model_name": "ConvNeXtTiny",
        "class_count": len(CLASS_NAMES),
        "classes": [DISPLAY_NAMES.get(name, name) for name in CLASS_NAMES],
        "error": model_error,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "Maximum image size is 10 MB.")
    if model is None:
        raise HTTPException(503, f"Model unavailable: {model_error}")

    probabilities = normalize(model.predict(prepare_image(raw), verbose=0)[0])
    indices = np.argsort(probabilities)[::-1]

    top = [
        PredictionItem(
            class_name=DISPLAY_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i]),
            confidence=float(probabilities[i]),
        )
        for i in indices[:3]
    ]

    return PredictionResponse(
        class_name=top[0].class_name,
        confidence=top[0].confidence,
        top_predictions=top,
        model="ConvNeXtTiny",
        demo=False,
    )
