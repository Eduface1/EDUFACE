from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Query, Path
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from deepface import DeepFace
from typing import List, Optional, Any, Dict
import shutil
import tempfile
import os
from datetime import date, datetime, timedelta
from .db import init_db, SessionLocal, Student, Attendance
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT


app = FastAPI(title="EDUFACE API", version="1.0.0")

# Configuración de reconocimiento (puedes ajustar por variables de entorno)
FACE_MODEL = os.getenv('FACE_MODEL', 'ArcFace')  # Opciones populares: ArcFace, Facenet512, VGG-Face
FACE_DETECTOR = os.getenv('FACE_DETECTOR', 'opencv')  # opencv es más rápido en CPU/Windows
# Umbral máximo de distancia aceptable y margen entre top1 y top2 (se ajustan según modelo)
FACE_MAX_DISTANCE = os.getenv('FACE_MAX_DISTANCE')  # si None, decidir según modelo
FACE_MIN_MARGIN = float(os.getenv('FACE_MIN_MARGIN', '0.04'))

def _defaults_for_model(model_name: str):
        name = (model_name or '').lower()
        if 'arc' in name:  # ArcFace
                return 'euclidean_l2', 1.20
        if 'facenet' in name:  # Facenet/Facenet512
                return 'cosine', 0.30
        # VGG-Face y otros
        return 'cosine', 0.30

# CORS (simple y abierto)
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

# Estáticos si existen
base_dir = os.path.dirname(__file__)
static_dir = os.path.join(base_dir, "static")
if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

# carpeta para fotos de estudiantes
students_media = os.path.abspath(os.path.join(base_dir, '..', 'students'))
os.makedirs(students_media, exist_ok=True)
app.mount('/media/students', StaticFiles(directory=students_media), name='students_media')


# DB utils
def get_db():
        db = SessionLocal()
        try:
                yield db
        finally:
                db.close()


@app.on_event("startup")
def _on_startup():
        init_db()


def _save_upload_to_temp(upload: UploadFile) -> str:
        suffix = os.path.splitext(upload.filename or "")[1]
        fd, path = tempfile.mkstemp(suffix=suffix or ".jpg")
        os.close(fd)
        with open(path, "wb") as f:
                shutil.copyfileobj(upload.file, f)
        return path


# Pydantic
class StudentBase(BaseModel):
        code: str
        name: str
        grade: Optional[str] = None
        section: Optional[str] = None
        gender: Optional[str] = None
        registration_date: Optional[date] = None
        photo_path: Optional[str] = None


class StudentCreate(StudentBase):
        pass


class StudentUpdate(BaseModel):
        name: Optional[str] = None
        grade: Optional[str] = None
        section: Optional[str] = None
        gender: Optional[str] = None
        registration_date: Optional[date] = None
        photo_path: Optional[str] = None


class StudentOut(StudentBase):
        id: int

        class Config:
                from_attributes = True


# DeepFace API básica
@app.post("/verify")
async def verify(img1: UploadFile = File(...), img2: UploadFile = File(...),
                                 model_name: str = Form("VGG-Face"), detector_backend: str = Form("opencv")):
        try:
                p1 = _save_upload_to_temp(img1)
                p2 = _save_upload_to_temp(img2)
                result = DeepFace.verify(img1_path=p1, img2_path=p2,
                                                                 model_name=model_name,
                                                                 detector_backend=detector_backend,
                                                                 enforce_detection=False)
                return JSONResponse(content=result)
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        finally:
                for p in [locals().get('p1'), locals().get('p2')]:
                        if p and os.path.exists(p):
                                os.remove(p)


@app.post("/recognize")
async def recognize(img: UploadFile = File(...), db_path: str = Form("db"),
                                        model_name: str = Form("VGG-Face"), detector_backend: str = Form("opencv"),
                                        distance_metric: str = Form("cosine"), threshold: Optional[float] = Form(None)):
        try:
                p = _save_upload_to_temp(img)
                if not os.path.isdir(db_path):
                        raise HTTPException(status_code=400, detail=f"db_path no existe: {db_path}")
                dfs = DeepFace.find(img_path=p, db_path=db_path,
                                                        model_name=model_name,
                                                        detector_backend=detector_backend,
                                                        distance_metric=distance_metric,
                                                        enforce_detection=False)
                df = dfs[0] if isinstance(dfs, list) else dfs
                if threshold is not None and df is not None and not df.empty and 'distance' in df.columns:
                        df = df[df['distance'] <= threshold]
                matches = df.to_dict(orient='records') if df is not None and not df.empty else []
                return {"matches": matches}
        except HTTPException:
                raise
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        finally:
                if locals().get('p') and os.path.exists(p):
                        os.remove(p)


@app.post("/analyze")
async def analyze(img: UploadFile = File(...),
                                    actions: List[str] = Form(["age", "gender", "emotion", "race"]),
                                    detector_backend: str = Form("opencv")):
        try:
                p = _save_upload_to_temp(img)
                result = DeepFace.analyze(img_path=p, actions=actions,
                                                                    detector_backend=detector_backend,
                                                                    enforce_detection=False)
                return {"results": result} if isinstance(result, list) else result
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        finally:
                if locals().get('p') and os.path.exists(p):
                        os.remove(p)


@app.get("/", response_class=HTMLResponse)
async def root():
        return """
        <!doctype html>
        <html>
        <head>
            <meta charset='utf-8'/>
            <meta name='viewport' content='width=device-width, initial-scale=1'/>
            <title>EDUFACE - Sistema de Asistencia Escolar</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
            :root{--brand:#6c56cf;--brand2:#e8e3ff}
            body{font-family:system-ui,Segoe UI,Roboto,Arial;margin:0;background:var(--brand2)}
            .wrap{max-width:1200px;margin:0 auto;padding:20px}
            header{background:var(--brand);color:#fff;padding:20px 16px;border-radius:0 0 12px 12px;box-shadow:0 6px 18px rgba(0,0,0,.18)}
            header h1{margin:0;font-size:32px;line-height:1.2;text-shadow:0 3px 10px rgba(0,0,0,.35)}
            header small{display:block;margin-top:4px;opacity:.95;font-size:18px;text-shadow:0 2px 8px rgba(0,0,0,.3)}
            .tabs{display:flex;gap:8px;margin:16px 0}
            .tab{background:#fff;color:#333;border-radius:10px;padding:10px 14px;cursor:pointer;border:2px solid transparent}
            .tab.active{border-color:var(--brand);box-shadow:0 1px 4px rgba(0,0,0,.08)}
            section.card{background:#fff;border-radius:12px;padding:16px;margin:16px 0;box-shadow:0 2px 6px rgba(0,0,0,.06)}
            .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
            @media(max-width:960px){.grid2{grid-template-columns:1fr}}
            .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
            @media(max-width:960px){.kpis{grid-template-columns:repeat(2,1fr)}}
            .kpi{background:#fff;border-radius:12px;padding:16px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.06)}
            .kpi .num{font-size:28px;font-weight:700}
            .row{display:flex;gap:8px;flex-wrap:wrap}
            input,select,button{padding:.5rem;border-radius:8px;border:1px solid #ccc}
            button{background:var(--brand);color:#fff;border:none}
            /* Cámara */
            .camBox{position:relative;width:520px;height:390px;background:#0e0e0e;border-radius:12px;overflow:visible;display:flex;align-items:center;justify-content:center}
            video{width:100%;height:100%;object-fit:cover;background:#000;border-radius:12px;display:block}
            .camOverlay{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#e5e5e5;text-align:center;gap:14px;padding:12px}
            .camTitle{font-size:18px;letter-spacing:.3px;opacity:.95}
            .camControls{display:flex;gap:18px}
            table{width:100%;border-collapse:collapse}
            th,td{border-bottom:1px solid #eee;padding:8px;text-align:left}

            /* Modal detalle */
            .modal{position:fixed;inset:0;background:rgba(0,0,0,.45);display:none;align-items:center;justify-content:center;padding:16px;z-index:1000}
            .modalContent{background:#fff;max-width:720px;width:100%;border-radius:12px;box-shadow:0 12px 28px rgba(0,0,0,.25);padding:16px;position:relative}
            .modalClose{position:absolute;top:8px;right:10px;background:#dc3545;color:#fff;border:none;border-radius:8px;padding:6px 10px;cursor:pointer;font-weight:bold;transition:background-color 0.2s}
            .modalClose:hover{background:#c82333}
            .cardItem{cursor:pointer}
            /* Toast */
            .toast{position:absolute;top:10px;left:50%;background:linear-gradient(135deg,#4b3bc9,#6c56cf);color:#fff;padding:12px 18px;border-radius:14px;box-shadow:0 14px 32px rgba(0,0,0,.32);opacity:0;transform:translate(-50%,0) scale(.98);transition:all .25s;z-index:5;border-left:6px solid #ffe082;display:flex;align-items:center;gap:10px;text-align:center;white-space:nowrap;max-width:none}
            .toast.show{opacity:1;transform:translate(-50%,0) scale(1)}
            .toast .tIcon{font-size:20px}
            .toast .tMsg{font-size:16px;line-height:1.2}

            /* Foto preview en formulario */
            .photoPreview{width:160px;height:160px;border-radius:12px;background:linear-gradient(135deg,#f2f2f2,#e6e6e6);display:flex;align-items:center;justify-content:center;color:#777;border:1px dashed #ccc;overflow:hidden}
            .photoPreview img{width:100%;height:100%;object-fit:cover;display:block}
            </style>
        </head>
        <body>
            <header>
                        <div class='wrap'>
                    <h1>🎓 EDUFACE - Sistema de Asistencia Escolar</h1>
                    <small>Reconocimiento facial automático para control de asistencia</small>
                </div>
            </header>
            <div class='wrap'>
                <div class='tabs'>
                  <div class='tab active' data-tab='realtime'>📹 Registro en Tiempo Real</div>
                    <div class='tab' data-tab='reports'>📊 Reportes Generales</div>
                    <div class='tab' data-tab='analysis'>📝 Análisis Detallado</div>
                  <div class='tab' data-tab='manage'>🧑‍🏫 Gestión de Estudiantes</div>
                </div>

                                <section id='realtime' class='card'>
                                        <h2>Reconocimiento Facial en Tiempo Real</h2>
                                        <div class='grid2'>
                                                <div>
                                                        <div class="camBox">
                                                                <video id="cam" autoplay playsinline></video>
                                                                <div id="camOverlay" class="camOverlay">
                                                                        <div class="camTitle">📹 Cámara de Reconocimiento Facial</div>
                                                                        <div class="camControls">
                                                                                <button id="btnStart" style='background:linear-gradient(135deg, #28a745 0%, #20c997 100%);color:#fff;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-weight:600;font-size:14px;transition:all 0.3s ease;box-shadow:0 2px 4px rgba(40,167,69,0.3);'>
                                                                                    ▶️ Iniciar Control
                                                                                </button>
                                                                        </div>
                                                                </div>
                                                                <div id="toast" class="toast"></div>
                                                        </div>
                                                        <div id="stopRow" class='row' style="justify-content:center;margin-top:8px;display:none">
                                                                <button id="btnStop" style='background:linear-gradient(135deg, #dc3545 0%, #c82333 100%);color:#fff;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-weight:600;font-size:14px;transition:all 0.3s ease;box-shadow:0 2px 4px rgba(220,53,69,0.3);'>
                                                                    ⏹️ Detener Control
                                                                </button>
                                                        </div>
                                                </div>
                                                <div>
                                                        <h3>Estudiantes Detectados Hoy</h3>
                                                        <div id="recentList" style="max-height:360px;overflow:auto"></div>
                                                </div>
                                        </div>
                                        <div class='kpis'>
                        <div class='kpi'><div>Total Hoy</div><div id='k_today' class='num'>-</div></div>
                        <div class='kpi'><div>Puntuales</div><div id='k_punctual_today' class='num'>-</div></div>
                        <div class='kpi'><div>Tarde</div><div id='k_late_today' class='num'>-</div></div>
                        <div class='kpi'><div>Ausentes</div><div id='k_absent_today' class='num'>-</div></div>
                    </div>
                </section>

                <section id='reports' class='card' style='display:none'>
                    <h2>Dashboard de Asistencia General</h2>
                    
                    <!-- Panel Superior: Filtros y KPIs -->
                    <div style='background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)'>
                        <div class='row' style='margin-bottom:16px'>
                            <label>Período:
                                <select id='period'>
                                    <option value='day'>Día</option>
                                    <option value='week'>Semana</option>
                                    <option value='month'>Mes</option>
                                </select>
                            </label>
                            <label>Grado:
                                <select id='grade'>
                                    <option value=''>Todos los grados</option>
                                    <option value='1'>1°</option>
                                    <option value='2'>2°</option>
                                    <option value='3'>3°</option>
                                    <option value='4'>4°</option>
                                    <option value='5'>5°</option>
                                    <option value='6'>6°</option>
                                </select>
                            </label>
                            <label>Sección:
                                <select id='section'>
                                    <option value=''>Todas las secciones</option>
                                    <option value='A'>A</option>
                                    <option value='B'>B</option>
                                    <option value='C'>C</option>
                                </select>
                            </label>
                            <button id='btnRefresh' style='background:linear-gradient(135deg, #28a745 0%, #20c997 100%);color:#fff;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-weight:600;font-size:14px;transition:all 0.3s ease;box-shadow:0 2px 4px rgba(40,167,69,0.3);'>
                                🔄 Actualizar
                            </button>
                        </div>
                        <div class='kpis'>
                            <div class='kpi'><div>Total estudiantes</div><div id='k_students' class='num'>-</div></div>
                            <div class='kpi'><div>Registros</div><div id='k_records' class='num'>-</div></div>
                            <div class='kpi'><div>Puntualidad</div><div id='k_punctual' class='num'>-</div></div>
                            <div class='kpi'><div>Tarde</div><div id='k_late' class='num'>-</div></div>
                            <div class='kpi'><div>Ausentes</div><div id='k_absent' class='num'>-</div></div>
                        </div>
                    </div>
                    
                    <!-- Panel Inferior: Gráficos -->
                    <div style='background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)'>
                        <div class='grid2'>
                            <div><h3>Tendencia de Asistencia Semanal</h3><canvas id='chWeekly' height='120'></canvas></div>
                            <div><h3>Asistencia por Grado (Hoy)</h3><canvas id='chGrade' height='120'></canvas></div>
                        </div>
                    </div>
                </section>

                <section id='analysis' class='card' style='display:none'>
                    <h2>Análisis Detallado</h2>
                    
                    <!-- Panel de Exportación PDF -->
                    <div style='background:#f8f9fa;border:1px solid #dee2e6;border-radius:12px;padding:20px;margin-bottom:24px;box-shadow:0 2px 6px rgba(0,0,0,.1)'>
                        <h3 style='margin:0 0 16px 0;color:#495057;display:flex;align-items:center;gap:8px;'>
                            <span>📋</span> Exportar Reporte de Asistencia
                        </h3>
                        <div class='grid2' style='gap:16px;margin-bottom:16px;'>
                            <div style='display:flex;flex-direction:column;gap:4px;'>
                                <label style='font-weight:600;color:#495057;font-size:14px;'>Fecha:</label>
                                <input type='date' id='exportDate' style='padding:10px;border:1px solid #ced4da;border-radius:6px;font-size:14px;'>
                            </div>
                            <div style='display:flex;flex-direction:column;gap:4px;'>
                                <label style='font-weight:600;color:#495057;font-size:14px;'>Grado:</label>
                                <select id='exportGrade' style='padding:10px;border:1px solid #ced4da;border-radius:6px;font-size:14px;background:#fff;'>
                                    <option value=''>Todos los grados</option>
                                    <option value='1'>1ro</option>
                                    <option value='2'>2do</option>
                                    <option value='3'>3ro</option>
                                    <option value='4'>4to</option>
                                    <option value='5'>5to</option>
                                    <option value='6'>6to</option>
                                </select>
                            </div>
                            <div style='display:flex;flex-direction:column;gap:4px;'>
                                <label style='font-weight:600;color:#495057;font-size:14px;'>Sección:</label>
                                <select id='exportSection' style='padding:10px;border:1px solid #ced4da;border-radius:6px;font-size:14px;background:#fff;'>
                                    <option value=''>Todas las secciones</option>
                                    <option value='A'>Sección A</option>
                                    <option value='B'>Sección B</option>
                                    <option value='C'>Sección C</option>
                                </select>
                            </div>
                            <div style='display:flex;flex-direction:column;gap:4px;'>
                                <label style='font-weight:600;color:#495057;font-size:14px;'>Estado:</label>
                                <select id='exportStatus' style='padding:10px;border:1px solid #ced4da;border-radius:6px;font-size:14px;background:#fff;'>
                                    <option value=''>Todos los estados</option>
                                    <option value='Puntual'>Puntuales</option>
                                    <option value='Tarde'>Tardanza</option>
                                    <option value='Ausente'>Ausentes</option>
                                </select>
                            </div>
                        </div>
                        <button id='btnExportPDF' style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);color:#fff;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-weight:600;font-size:14px;transition:all 0.3s ease;box-shadow:0 2px 4px rgba(102,126,234,0.3);'>
                            📄 Generar Reporte PDF
                        </button>
                    </div>
                    
                    <div class='grid2'>
                        <div><h3>Distribución por Género</h3><canvas id='chGender' height='120'></canvas></div>
                        <div><h3>Patrones de Llegada por Hora</h3><canvas id='chArrival' height='120'></canvas></div>
                    </div>
                    <h3>Registro Detallado</h3>
                    <table>
                        <thead><tr><th>Estudiante</th><th>Grado/Sección</th><th>Fecha</th><th>Hora</th><th>Estado</th></tr></thead>
                        <tbody id='tblRecent'></tbody>
                    </table>
                </section>

        

                                        <section id='manage' class='card' style='display:none'>
                                                <div class='row' style='align-items:center;gap:12px;'>
                                                        <h2 style='margin:0'>Gestión de Estudiantes</h2>
                                                        <button id='btnAddStudent' style='background:linear-gradient(135deg, #007bff 0%, #0056b3 100%);color:#fff;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-weight:600;font-size:14px;transition:all 0.3s ease;box-shadow:0 2px 4px rgba(0,123,255,0.3);'>
                                                            ➕ Agregar Estudiante
                                                        </button>
                                                </div>
                                                <h3 style='margin-top:12px'>Lista de Estudiantes</h3>
                                                <table>
                                                        <thead><tr><th>Nombre</th><th>Grado</th><th>Sección</th><th>Género</th><th>Fecha Registro</th><th>Foto</th><th></th></tr></thead>
                                                        <tbody id='tblStudents'></tbody>
                                                </table>
                                        </section>

            </div>

            <script>
                // Tabs
                document.querySelectorAll('.tab').forEach(t=>{
                    t.onclick = ()=>{
                        document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
                        t.classList.add('active');
                        const key = t.dataset.tab;
                            ['realtime','reports','analysis','manage'].forEach(id=>{
                            document.getElementById(id).style.display = (id===key)?'block':'none';
                        });
                                                // Actualizar contenido al entrar a cada apartado
                                                if(key==='realtime'){
                                                        refreshToday();
                                                        refreshTodayKPIs();
                                                } else if(key==='reports'){
                                                        loadSummary();
                                                        loadWeekly();
                                                        loadByGrade();
                                                } else if(key==='analysis'){
                                                        loadDetailed();
                                                } else if(key==='manage'){
                                                        loadStudents();
                                                }
                    }
                });

                // Cámara y tiempo real
                                let stream, timer, running=false, busy=false;
                                const cam = document.getElementById('cam');
                                const camOverlay = document.getElementById('camOverlay');
                                const stopRow = document.getElementById('stopRow');
                const recentList = document.getElementById('recentList');
                                async function startCam(){
                                        stream = await navigator.mediaDevices.getUserMedia({video:true});
                                        cam.srcObject = stream;
                                        if (camOverlay) camOverlay.style.display = 'none';
                                }
                async function capture(){
                    const cvs = document.createElement('canvas');
                                        // Redimensionar a menor tamaño para acelerar DeepFace
                                        const scaleW = 360; // ancho objetivo
                                        const ratio = (cam.videoWidth||520) / (cam.videoHeight||390);
                                        const w = scaleW;
                                        const h = Math.round(scaleW / ratio);
                                        cvs.width = w; cvs.height = h;
                    const ctx = cvs.getContext('2d');
                                        ctx.drawImage(cam, 0, 0, w, h);
                    const blob = await new Promise(res=>cvs.toBlob(res,'image/jpeg',0.85));
                    const fd = new FormData();
                    fd.append('img', blob, 'frame.jpg');
                    fd.append('db_path', 'db');
                                                                const resp = await fetch('/attendance/mark', {method:'POST', body: fd});
                                                                try{
                                                                        const j = await resp.json();
                                                                        if(j && j.status==='marked' && j.student){ showToast(j.student); }
                                                                }catch{}
                                        refreshToday(); refreshTodayKPIs();
                }
                                async function loop(){
                                        if(!running) return;
                                        if(busy){ timer = setTimeout(loop, 300); return; }
                                        busy = true;
                                        try { await capture(); } finally { busy = false; }
                                        timer = setTimeout(loop, 4500); // siguiente captura tras ~4.5s
                                }
                                document.getElementById('btnStart').onclick = async ()=>{
                                        if(running) return;
                                        await startCam();
                                        running = true;
                                        if (stopRow) stopRow.style.display = 'flex';
                                        // Primera captura inmediata para reducir latencia percibida
                                        loop();
                                };
                                document.getElementById('btnStop').onclick = ()=>{
                                        running = false;
                                        if(timer) { clearTimeout(timer); timer = null; }
                                        if(stream){ stream.getTracks().forEach(t=>t.stop()); }
                                        cam.srcObject = null; stream = null;
                                        if (camOverlay) camOverlay.style.display = 'flex';
                                        if (stopRow) stopRow.style.display = 'none';
                                };

                                                                                function pill(status){ return status==='Puntual' ? '<span style="background:#c8f7c5;color:#2b7a2b;padding:2px 8px;border-radius:999px;font-size:12px">✔ Puntual</span>' : '<span style="background:#ffe7c2;color:#8a5300;padding:2px 8px;border-radius:999px;font-size:12px">⚠ Tarde</span>'; }
                                        async function refreshToday(){
                                                const r = await fetch('/attendance/today');
                                                const j = await r.json();
                                                                                                recentList.innerHTML = j.map(i=>{
                                                        const grado = (i.grade||'') + (i.section?(' ' + i.section):'');
                                                                                                                return `<div class="cardItem" data-id="${i.student_id}" style="border:1px solid #e5e5e5;border-radius:10px;padding:10px;margin:8px 0;box-shadow:0 1px 2px rgba(0,0,0,.03)">
                                                                <div style="font-weight:700">${i.name}</div>
                                                                <div style="font-size:13px;color:#555">Grado: ${grado||'-'}</div>
                                                                <div style="font-size:13px;color:#555">Hora: ${i.time}</div>
                                                                <div style="font-size:13px;color:#555">Fecha: ${i.date}</div>
                                                                <div style="margin-top:6px">${pill(i.status)}</div>
                                                        </div>`
                                                }).join('');
                                                                                                // Delegación de clic para abrir modal con historial
                                                                                                recentList.querySelectorAll('.cardItem').forEach(el=>{
                                                                                                        el.onclick = async ()=>{
                                                                                                                const id = el.getAttribute('data-id');
                                                                                                                try{
                                                                                                                        const [sRes, hRes] = await Promise.all([
                                                                                                                                fetch(`/students/${id}`),
                                                                                                                                fetch(`/students/${id}/attendance?limit=20`)
                                                                                                                        ]);
                                                                                                                        if(!sRes.ok || !hRes.ok) throw new Error('Error al cargar historial');
                                                                                                                        const s = await sRes.json();
                                                                                                                        const hist = await hRes.json();
                                                                                                                        openModal(renderHistory(s, hist));
                                                                                                                }catch(err){ alert(err.message||err); }
                                                                                                        }
                                                                                                })
                                        }
                async function refreshTodayKPIs(){
                    const r = await fetch('/metrics/today');
                    const j = await r.json();
                    k_today.textContent = j.total_today;
                    k_punctual_today.textContent = j.punctual_today;
                    k_late_today.textContent = j.late_today;
                    k_absent_today.textContent = j.absent_today;
                }

                // Reportes generales
                let ch1, ch2, ch3, ch4;
                async function loadSummary(){
                    const period = document.getElementById('period').value;
                    const grade = document.getElementById('grade').value;
                    const section = document.getElementById('section').value;
                    
                    const params = new URLSearchParams();
                    if (period) params.append('period', period);
                    if (grade) params.append('grade', grade);
                    if (section) params.append('section', section);
                    
                    const r = await fetch(`/metrics/summary?${params.toString()}`);
                    const j = await r.json();
                    k_students.textContent = j.students_total;
                    k_records.textContent = j.records_total;
                    k_punctual.textContent = j.punctuality_pct + '%';
                    k_late.textContent = j.late_pct + '%';
                    k_absent.textContent = j.absent_pct + '%';
                }
                async function loadWeekly(){
                    const r = await fetch('/metrics/weekly');
                    const j = await r.json();
                    const ctx = document.getElementById('chWeekly').getContext('2d');
                    if(ch1) ch1.destroy();
                    ch1 = new Chart(ctx,{type:'line',data:{labels:j.labels,datasets:[{label:'Asistencia %',data:j.attendance_pct,fill:true,borderColor:'#3f51b5',backgroundColor:'rgba(63,81,181,.1)'}]}});
                }
                async function loadByGrade(){
                    const r = await fetch('/metrics/by-grade');
                    const j = await r.json();
                    const labels = Object.keys(j);
                    const data = Object.values(j);
                    const ctx = document.getElementById('chGrade').getContext('2d');
                    if(ch2) ch2.destroy();
                    ch2 = new Chart(ctx,{type:'bar',data:{labels,datasets:[{label:'Asistencia (hoy)',data,backgroundColor:'#7ea1ff'}]}});
                }
                btnRefresh.onclick = ()=>{ loadSummary(); loadWeekly(); loadByGrade(); };

                // Análisis detallado
                async function loadDetailed(){
                    const r = await fetch('/metrics/detailed');
                    const j = await r.json();
                    const gLabels = Object.keys(j.gender_distribution);
                    const gData = Object.values(j.gender_distribution);
                    const cg = document.getElementById('chGender').getContext('2d');
                    if(ch3) ch3.destroy();
                    ch3 = new Chart(cg,{type:'doughnut',data:{labels:gLabels,datasets:[{data:gData,backgroundColor:['#80cbc4','#64b5f6','#ffe082','#e57373']}]}});
                    const aLabels = Object.keys(j.arrival_buckets);
                    const aData = Object.values(j.arrival_buckets);
                    const ca = document.getElementById('chArrival').getContext('2d');
                    if(ch4) ch4.destroy();
                    ch4 = new Chart(ca,{type:'bar',data:{labels:aLabels,datasets:[{label:'Cantidad de estudiantes',data:aData,backgroundColor:'#9ccc65'}]}});
                    const r2 = await fetch('/attendance/recent?limit=10');
                    const rows = await r2.json();
                    tblRecent.innerHTML = rows.map(x=>`<tr><td>${x.name}</td><td>${x.grade||''} ${x.section||''}</td><td>${x.date}</td><td>${x.time}</td><td>${x.status}</td></tr>`).join('');
                }

                                        // Gestión de estudiantes
                                        async function loadStudents(){
                                                const r = await fetch('/students');
                                                const j = await r.json();
                                                tblStudents.innerHTML = j.map(s=>{
                                                        const img = s.photo_path ? `<img src="${s.photo_path}" alt="foto" style="width:40px;height:40px;object-fit:cover;border-radius:6px"/>` : '';
                                                        const dreg = s.registration_date || '';
                                                        return `<tr>
                                                                <td>${s.name}</td>
                                                                <td>${s.grade||''}</td>
                                                                <td>${s.section||''}</td>
                                                                <td>${s.gender||''}</td>
                                                                <td>${dreg}</td>
                                                                <td>${img}</td>
                                                                <td><button onclick="editStudent(${s.id})" style='background:linear-gradient(135deg, #ffc107 0%, #e0a800 100%);color:#000;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-weight:600;font-size:12px;transition:all 0.3s ease;box-shadow:0 2px 4px rgba(255,193,7,0.3);'>✏️ Editar</button></td>
                                                        </tr>`
                                                }).join('');
                                        }
                                        // Botón agregar: abre modal con formulario
                                        document.getElementById('btnAddStudent').onclick = ()=> openStudentForm();

                                        // Render de formulario en modal
                                        function studentFormHTML(mode, st){
                                                const isEdit = mode==='edit';
                                                const title = isEdit ? 'Editar Estudiante' : 'Agregar Estudiante';
                                                const name = isEdit ? (st.name||'') : '';
                                                const grade = isEdit ? (st.grade||'') : '';
                                                const section = isEdit ? (st.section||'') : '';
                                                const gender = isEdit ? (st.gender||'') : '';
                                                const registration_date = isEdit ? (st.registration_date||'') : '';
                                                const photo = isEdit ? (st.photo_path||'') : '';
                                                return `
                                                <h3 style="margin-top:0">${title}</h3>
                                                <form id="studentForm" enctype="multipart/form-data">
                                                        <div class="row" style="align-items:flex-end">
                                                                <div>
                                                                        <label>Nombre<br/><input name="name" required value="${name}"/></label>
                                                                </div>
                                                                <div>
                                                                        <label>Grado<br/><input name="grade" value="${grade}"/></label>
                                                                </div>
                                                                <div>
                                                                        <label>Sección<br/><input name="section" value="${section}"/></label>
                                                                </div>
                                                                <div>
                                                                        <label>Género<br/>
                                                                                <select name="gender">
                                                                                        <option value="" ${gender===''?'selected':''}>-</option>
                                                                                        <option ${gender==='Femenino'?'selected':''}>Femenino</option>
                                                                                        <option ${gender==='Masculino'?'selected':''}>Masculino</option>
                                                                                </select>
                                                                        </label>
                                                                </div>
                                                                <div>
                                                                        <label>Fecha de Registro<br/><input type="date" name="registration_date" value="${registration_date}"/></label>
                                                                </div>
                                                        </div>
                                                        <div class="row" style="margin-top:10px;align-items:center">
                                                                <div>
                                                                        <div id="photoPreview" class="photoPreview">${photo?`<img src="${photo}" alt="foto"/>`:'Foto'}</div>
                                                                </div>
                                                                <div>
                                                                        <label>Foto<br/><input id="photoInput" type="file" name="photo" accept="image/*"/></label>
                                                                        <div style="font-size:12px;color:#666">Se mostrará una vista previa</div>
                                                                </div>
                                                        </div>
                                                        <div class="row" style="margin-top:12px">
                                                                <button type="submit" style='background:linear-gradient(135deg, #28a745 0%, #20c997 100%);color:#fff;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-weight:600;font-size:14px;transition:all 0.3s ease;box-shadow:0 2px 4px rgba(40,167,69,0.3); margin-right: 8px;'>
                                                                    💾 Guardar
                                                                </button>
                                                                <button type="button" onclick="closeModal()" style="background:linear-gradient(135deg, #6c757d 0%, #495057 100%);color:#fff;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-weight:600;font-size:14px;transition:all 0.3s ease;box-shadow:0 2px 4px rgba(108,117,125,0.3); margin-right: 8px;">
                                                                    ✖️ Cancelar
                                                                </button>
                                                                ${isEdit?'<button type="button" id="btnDelete" style="background:linear-gradient(135deg, #dc3545 0%, #c82333 100%);color:#fff;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-weight:600;font-size:14px;transition:all 0.3s ease;box-shadow:0 2px 4px rgba(220,53,69,0.3);">🗑️ Eliminar</button>':''}
                                                        </div>
                                                </form>`
                                        }

                                        function openStudentForm(st){
                                                const mode = st? 'edit' : 'add';
                                                openModal(studentFormHTML(mode, st||{}));
                                                const form = document.getElementById('studentForm');
                                                const photoInput = document.getElementById('photoInput');
                                                const preview = document.getElementById('photoPreview');
                                                if(photoInput){
                                                        photoInput.addEventListener('change', ev=>{
                                                                const f = ev.target.files && ev.target.files[0];
                                                                if(!f){ preview.innerHTML = 'Foto'; return; }
                                                                const fr = new FileReader();
                                                                fr.onload = ()=>{ preview.innerHTML = `<img src="${fr.result}"/>`; };
                                                                fr.readAsDataURL(f);
                                                        });
                                                }
                                                form.addEventListener('submit', async (e)=>{
                                                        e.preventDefault();
                                                        const fd = new FormData(form);
                                                        if(mode==='add'){
                                                                const name = fd.get('name');
                                                                const code = (name||'').toString().trim().toLowerCase().replace(/\s+/g,'_').replace(/[^a-z0-9_]/g,'');
                                                                fd.append('code', code || ('alumno_' + Date.now()));
                                                                const r = await fetch('/students/form', { method:'POST', body: fd });
                                                                const t = await r.text();
                                                                if(!r.ok){ alert(t); return; }
                                                        } else {
                                                                const r = await fetch(`/students/${st.id}/form`, { method:'PUT', body: fd });
                                                                const t = await r.text();
                                                                if(!r.ok){ alert(t); return; }
                                                        }
                                                        closeModal();
                                                        loadStudents();
                                                });
                                                if(st && document.getElementById('btnDelete')){
                                                        document.getElementById('btnDelete').onclick = async ()=>{
                                                                if(!confirm('¿Eliminar estudiante?')) return;
                                                                const r = await fetch(`/students/${st.id}`, { method:'DELETE' });
                                                                if(!r.ok){ alert(await r.text()); return; }
                                                                closeModal();
                                                                loadStudents();
                                                        }
                                                }
                                        }

                                        async function editStudent(id){
                                                const st = await (await fetch('/students/'+id)).json();
                                                openStudentForm(st);
                                        }

                // Función para exportar PDF
                async function exportToPDF() {
                    const date = document.getElementById('exportDate').value;
                    const grade = document.getElementById('exportGrade').value;
                    const section = document.getElementById('exportSection').value;
                    const status = document.getElementById('exportStatus').value;
                    
                    if (!date) {
                        alert('Por favor selecciona una fecha para generar el reporte.');
                        return;
                    }
                    
                    // Preparar parámetros de filtro
                    const params = new URLSearchParams();
                    params.append('date', date);
                    if (grade) params.append('grade', grade);
                    if (section) params.append('section', section);
                    if (status) params.append('status', status);
                    
                    try {
                        document.getElementById('btnExportPDF').innerHTML = '⏳ Generando PDF...';
                        document.getElementById('btnExportPDF').disabled = true;
                        
                        const response = await fetch(`/export/pdf?${params.toString()}`);
                        
                        if (!response.ok) {
                            throw new Error('Error al generar el PDF');
                        }
                        
                        // Descargar el archivo
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `reporte_asistencia_${date}.pdf`;
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        document.body.removeChild(a);
                        
                    } catch (error) {
                        console.error('Error:', error);
                        alert('Error al generar el PDF. Intenta nuevamente.');
                    } finally {
                        document.getElementById('btnExportPDF').innerHTML = '📄 Generar Reporte PDF';
                        document.getElementById('btnExportPDF').disabled = false;
                    }
                }
                
                // Configurar fecha por defecto a hoy
                document.getElementById('exportDate').value = new Date().toISOString().split('T')[0];
                
                // Vincular evento del botón
                document.getElementById('btnExportPDF').onclick = exportToPDF;

                // Cargas iniciales
                refreshToday(); refreshTodayKPIs();
                loadSummary(); loadWeekly(); loadByGrade();
                loadDetailed();
                loadStudents();

                                // Modal helpers
                                function openModal(html){
                                        const m = document.getElementById('modal');
                                        const c = document.getElementById('modalBody');
                                        c.innerHTML = html;
                                        m.style.display = 'flex';
                                }
                                function closeModal(){ document.getElementById('modal').style.display = 'none'; }
                                function renderHistory(s, hist){
                                        const img = s.photo_path ? `<img src="${s.photo_path}" alt="foto" style="width:56px;height:56px;object-fit:cover;border-radius:8px;margin-right:10px"/>` : '';
                                        const head = `<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">${img}<div><div style="font-weight:700;font-size:18px">${s.name}</div><div style="font-size:13px;color:#555">${s.grade||''} ${s.section||''}</div></div></div>`;
                                        const rows = hist.map(x=>`<tr><td>${x.date}</td><td>${x.time}</td><td>${x.status}</td></tr>`).join('');
                                        return `${head}
                                                <table style="width:100%;border-collapse:collapse">
                                                        <thead><tr><th style="text-align:left;border-bottom:1px solid #eee;padding:6px">Fecha</th><th style="text-align:left;border-bottom:1px solid #eee;padding:6px">Hora</th><th style="text-align:left;border-bottom:1px solid #eee;padding:6px">Estado</th></tr></thead>
                                                        <tbody>${rows || '<tr><td colspan=3 style="padding:8px">Sin registros</td></tr>'}</tbody>
                                                </table>`;
                                }

                                                                // Toast
                                                                let toastTimer;
                                                                function titleCase(str){
                                                                        return (str||'').toString().trim().split(/\s+/).map(w=> w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
                                                                }
                                                                function showToast(codeOrName){
                                                                        const raw = (codeOrName||'').toString().replaceAll('_',' ');
                                                                        const name = titleCase(raw);
                                                                        const el = document.getElementById('toast');
                                                                        el.innerHTML = `<span class="tIcon">🎉</span><span class="tMsg">Bienvenido <b>${name}</b>, ¡que tengas un bonito día!</span>`;
                                                                        el.classList.add('show');
                                                                        if(toastTimer){ clearTimeout(toastTimer); }
                                                                        toastTimer = setTimeout(()=>{ el.classList.remove('show'); }, 3600);
                                                                }
            </script>
                        <!-- Modal de detalle de estudiante -->
                        <div id="modal" class="modal">
                                <div class="modalContent">
                                        <button class="modalClose" onclick="closeModal()">✕</button>
                                        <div id="modalBody"></div>
                                </div>
                        </div>
        </body>
        </html>
        """


# Asistencia: marcar por rostro
@app.post('/attendance/mark')
async def attendance_mark(img: UploadFile = File(...),
                                                    db_path: str = Form('db'),
                                                    detector_backend: str = Form(FACE_DETECTOR),
                                                    model_name: str = Form(FACE_MODEL),
                                                    max_distance: Optional[float] = Form(None),
                                                    min_margin: float = Form(FACE_MIN_MARGIN),
                                                    db: Session = Depends(get_db)):
        p = _save_upload_to_temp(img)
        try:
                # Elegir métrica y umbral por defecto según modelo si no vienen forzados
                metric_default, threshold_default = _defaults_for_model(model_name)
                eff_max_distance = float(max_distance) if max_distance is not None else float(FACE_MAX_DISTANCE) if FACE_MAX_DISTANCE else threshold_default

                dfs = DeepFace.find(img_path=p, db_path=db_path,
                                                        model_name=model_name,
                                                        detector_backend=detector_backend,
                                                        distance_metric=metric_default,
                                                        enforce_detection=False)
                df = dfs[0] if isinstance(dfs, list) else dfs
                if df is None or df.empty:
                        return {"status": "unknown"}
                df = df.sort_values('distance')
                best = df.iloc[0]
                best_dist = float(best['distance'])
                second_dist = float(df.iloc[1]['distance']) if len(df) > 1 else None
                # Reglas: debe pasar el umbral absoluto y además tener separación suficiente del segundo mejor
                if best_dist > eff_max_distance:
                        print(f"[recognition] decision=unknown reason=above_threshold dist={best_dist:.4f} thr={eff_max_distance:.4f} model={model_name} det={detector_backend}")
                        return {"status": "unknown", "reason": "above_threshold", "distance": best_dist}
                if second_dist is not None and (second_dist - best_dist) < min_margin:
                        print(f"[recognition] decision=unknown reason=low_margin best={best_dist:.4f} second={second_dist:.4f} margin={(second_dist-best_dist):.4f} min_margin={min_margin:.4f}")
                        return {"status": "unknown", "reason": "low_margin", "distance": best_dist, "margin": second_dist - best_dist}
                identity_path = best['identity']
                code = os.path.basename(os.path.dirname(identity_path))

                student = db.query(Student).filter_by(code=code).one_or_none()
                if not student:
                        student = Student(code=code, name=code)
                        db.add(student)
                        db.commit()
                        db.refresh(student)

                today = date.today()
                att = db.query(Attendance).filter_by(student_id=student.id, date=today).one_or_none()
                if not att:
                        now = datetime.now()
                        cutoff = now.replace(hour=7, minute=30, second=0, microsecond=0)
                        status = 'Tarde' if now > cutoff else 'Puntual'
                        att = Attendance(student_id=student.id, date=today, time=now, status=status)
                        db.add(att)
                        db.commit()
                        db.refresh(att)
                print(f"[recognition] decision=marked code={code} dist={best_dist:.4f} model={model_name} det={detector_backend}")
                return {"status": "marked", "student": student.code, "attendance": {"date": str(att.date), "time": att.time.isoformat(), "status": att.status}, "distance": best_dist, "model": model_name, "detector": detector_backend}
        finally:
                if os.path.exists(p):
                        os.remove(p)


# Listados y métricas
@app.get('/attendance/today')
def attendance_today(db: Session = Depends(get_db)):
        today = date.today()
        rows = (
                db.query(Attendance, Student)
                .join(Student, Student.id == Attendance.student_id)
                .filter(Attendance.date == today)
                .order_by(Attendance.time.desc())
                .all()
        )
        out: List[Dict[str, Any]] = []
        for att, st in rows:
                out.append({
                        'student_id': st.id,
                        'code': st.code,
                        'name': st.name,
                        'grade': st.grade,
                        'section': st.section,
                        'gender': st.gender,
                        'date': str(att.date),
                        'time': att.time.strftime('%H:%M:%S'),
                        'status': att.status,
                })
        return out


# Endpoint temporal para limpiar asistencias de hoy (solo uso de pruebas)
@app.post('/admin/attendance/clear-today')
def admin_clear_today(db: Session = Depends(get_db)):
        today = date.today()
        deleted = db.query(Attendance).filter(Attendance.date == today).delete(synchronize_session=False)
        db.commit()
        return {"deleted": deleted, "date": str(today)}


@app.post('/admin/attendance/clear-all')
def admin_clear_all(db: Session = Depends(get_db)):
        deleted = db.query(Attendance).delete(synchronize_session=False)
        db.commit()
        return {"deleted": deleted}


@app.post('/admin/attendance/clear-date')
def admin_clear_date(target: str = Form(...), db: Session = Depends(get_db)):
        try:
                d = datetime.strptime(target, '%Y-%m-%d').date()
        except Exception:
                raise HTTPException(400, 'Fecha inválida, use YYYY-MM-DD')
        deleted = db.query(Attendance).filter(Attendance.date == d).delete(synchronize_session=False)
        db.commit()
        return {"deleted": deleted, "date": str(d)}


# Alias GET para pruebas rápidas en navegador
@app.get('/admin/attendance/clear-today')
def admin_clear_today_get(db: Session = Depends(get_db)):
        return admin_clear_today(db)


@app.get('/admin/attendance/clear-all')
def admin_clear_all_get(db: Session = Depends(get_db)):
        return admin_clear_all(db)


@app.get('/admin/attendance/counts')
def admin_counts(db: Session = Depends(get_db)):
        today = date.today()
        total = db.query(Attendance).count()
        today_count = db.query(Attendance).filter(Attendance.date == today).count()
        return {"total": total, "today": today_count, "date": str(today)}


@app.get('/attendance/recent')
def attendance_recent(limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)):
        rows = (
                db.query(Attendance, Student)
                .join(Student, Student.id == Attendance.student_id)
                .order_by(Attendance.time.desc())
                .limit(limit)
                .all()
        )
        out: List[Dict[str, Any]] = []
        for att, st in rows:
                out.append({
                        'name': st.name,
                        'grade': st.grade,
                        'section': st.section,
                        'date': str(att.date),
                        'time': att.time.strftime('%H:%M:%S'),
                        'status': att.status,
                })
        return out


@app.get('/metrics/today')
def metrics_today(db: Session = Depends(get_db)):
        today = date.today()
        total_students = db.query(Student).count()
        q = db.query(Attendance).filter(Attendance.date == today)
        total_today = q.count()
        punctual = q.filter(Attendance.status == 'Puntual').count()
        late = q.filter(Attendance.status == 'Tarde').count()
        absent = max(total_students - total_today, 0)
        return {
                'total_today': total_today,
                'punctual_today': punctual,
                'late_today': late,
                'absent_today': absent,
        }


@app.get('/metrics/summary')
def metrics_summary(
        period: Optional[str] = Query(None, enum=['day', 'week', 'month']),
        grade: Optional[str] = Query(None),
        section: Optional[str] = Query(None),
        start: Optional[date] = Query(None), 
        end: Optional[date] = Query(None), 
        db: Session = Depends(get_db)):
        
        # Definir rango de fechas según período
        if period and not start and not end:
                today = date.today()
                if period == 'day':
                        start = end = today
                elif period == 'week':
                        # Semana actual (lunes a domingo)
                        days_since_monday = today.weekday()
                        start = today - timedelta(days=days_since_monday)
                        end = start + timedelta(days=6)
                elif period == 'month':
                        # Mes actual
                        start = today.replace(day=1)
                        if today.month == 12:
                                end = date(today.year + 1, 1, 1) - timedelta(days=1)
                        else:
                                end = date(today.year, today.month + 1, 1) - timedelta(days=1)
        
        # Query base para attendance con join de student
        q = db.query(Attendance).join(Student, Student.id == Attendance.student_id)
        
        # Filtrar por fechas
        if start:
                q = q.filter(Attendance.date >= start)
        if end:
                q = q.filter(Attendance.date <= end)
        
        # Filtrar por grado y sección
        if grade:
                q = q.filter(Student.grade == grade)
        if section:
                q = q.filter(Student.section == section)
        
        total = q.count()
        punctual = q.filter(Attendance.status == 'Puntual').count()
        late = q.filter(Attendance.status == 'Tarde').count()
        
        # Query para estudiantes totales (aplicando filtros de grado/sección)
        student_q = db.query(Student)
        if grade:
                student_q = student_q.filter(Student.grade == grade)
        if section:
                student_q = student_q.filter(Student.section == section)
        students_total = student_q.count()
        
        punctuality = (punctual / total * 100.0) if total else 0.0
        late_pct = (late / total * 100.0) if total else 0.0
        
        # Calcular ausentes: estudiantes registrados sin asistencia en el período
        if start and end:
            # Para un rango de fechas específico, calcular días únicos con asistencia
            unique_students_with_attendance = db.query(Attendance.student_id).join(Student, Student.id == Attendance.student_id)
            if start:
                unique_students_with_attendance = unique_students_with_attendance.filter(Attendance.date >= start)
            if end:
                unique_students_with_attendance = unique_students_with_attendance.filter(Attendance.date <= end)
            if grade:
                unique_students_with_attendance = unique_students_with_attendance.filter(Student.grade == grade)
            if section:
                unique_students_with_attendance = unique_students_with_attendance.filter(Student.section == section)
            
            students_with_attendance = unique_students_with_attendance.distinct().count()
            absent_students = students_total - students_with_attendance
            absent_pct = (absent_students / students_total * 100.0) if students_total else 0.0
        else:
            # Para todos los registros, los ausentes son los que nunca tuvieron asistencia
            students_with_any_attendance = db.query(Attendance.student_id).join(Student, Student.id == Attendance.student_id)
            if grade:
                students_with_any_attendance = students_with_any_attendance.filter(Student.grade == grade)
            if section:
                students_with_any_attendance = students_with_any_attendance.filter(Student.section == section)
            
            students_with_attendance = students_with_any_attendance.distinct().count()
            absent_students = students_total - students_with_attendance
            absent_pct = (absent_students / students_total * 100.0) if students_total else 0.0
        
        return {
                "students_total": students_total,
                "records_total": total,
                "punctuality_pct": round(punctuality, 2),
                "late_pct": round(late_pct, 2),
                "absent_pct": round(absent_pct, 2),
        }


@app.get('/metrics/weekly')
def metrics_weekly(db: Session = Depends(get_db)):
        # Últimos 5 días (L-V) o últimos 7 días si prefieres
        labels: List[str] = []
        attendance_pct: List[float] = []
        total_students = db.query(Student).count()
        for i in range(6, -1, -1):
                d = date.today() - timedelta(days=i)
                labels.append(d.strftime('%a'))
                total = db.query(Attendance).filter(Attendance.date == d).count()
                pct = (total / total_students * 100.0) if total_students else 0.0
                attendance_pct.append(round(pct, 2))
        return {"labels": labels, "attendance_pct": attendance_pct}


@app.get('/metrics/by-grade')
def metrics_by_grade(db: Session = Depends(get_db)):
        today = date.today()
        rows = (
                db.query(Attendance, Student)
                .join(Student, Student.id == Attendance.student_id)
                .filter(Attendance.date == today)
                .all()
        )
        agg: Dict[str, int] = {}
        for att, st in rows:
                key = st.grade or 'Sin grado'
                agg[key] = agg.get(key, 0) + 1
        return agg


@app.get('/metrics/detailed')
def metrics_detailed(start: Optional[date] = Query(None), end: Optional[date] = Query(None), grade: Optional[str] = None, section: Optional[str] = None, db: Session = Depends(get_db)):
        q = db.query(Attendance, Student).join(Student, Student.id == Attendance.student_id)
        if start:
                q = q.filter(Attendance.date >= start)
        if end:
                q = q.filter(Attendance.date <= end)
        if grade:
                q = q.filter(Student.grade == grade)
        if section:
                q = q.filter(Student.section == section)
        rows = q.all()

        gender_counts: Dict[str, int] = {}
        buckets: Dict[str, int] = {'7:30-7:45': 0, '7:45-8:00': 0, '8:00-8:15': 0, '8:15-8:30': 0, '8:30+': 0}
        for att, st in rows:
                g = (st.gender or '-')
                gender_counts[g] = gender_counts.get(g, 0) + 1
                t = att.time.time()
                if t >= datetime.strptime('07:30','%H:%M').time() and t < datetime.strptime('07:45','%H:%M').time():
                        buckets['7:30-7:45'] += 1
                elif t < datetime.strptime('08:00','%H:%M').time():
                        buckets['7:45-8:00'] += 1
                elif t < datetime.strptime('08:15','%H:%M').time():
                        buckets['8:00-8:15'] += 1
                elif t < datetime.strptime('08:30','%H:%M').time():
                        buckets['8:15-8:30'] += 1
                else:
                        buckets['8:30+'] += 1

        return {'gender_distribution': gender_counts, 'arrival_buckets': buckets, 'count': len(rows)}


# CRUD Estudiantes
def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)


@app.post('/students', response_model=StudentOut)
def create_student(payload: StudentCreate, db: Session = Depends(get_db)):
        existing = db.query(Student).filter_by(code=payload.code).one_or_none()
        if existing:
                raise HTTPException(400, detail='Código ya existe')
        st = Student(
                code=payload.code,
                name=payload.name,
                grade=payload.grade,
                section=payload.section,
                gender=payload.gender,
                registration_date=payload.registration_date,
                photo_path=payload.photo_path,
        )
        db.add(st)
        db.commit()
        db.refresh(st)
        return st


@app.get('/students', response_model=List[StudentOut])
def list_students(grade: Optional[str] = None, section: Optional[str] = None, q: Optional[str] = None, db: Session = Depends(get_db)):
        query = db.query(Student)
        if grade:
                query = query.filter(Student.grade == grade)
        if section:
                query = query.filter(Student.section == section)
        if q:
                like = f"%{q}%"
                query = query.filter((Student.name.like(like)) | (Student.code.like(like)))
        return query.order_by(Student.name).all()


@app.get('/students/{student_id}', response_model=StudentOut)
def get_student(student_id: int = Path(..., ge=1), db: Session = Depends(get_db)):
        st = db.get(Student, student_id)
        if not st:
                raise HTTPException(404, 'No encontrado')
        return st


@app.put('/students/{student_id}', response_model=StudentOut)
def update_student(student_id: int, payload: StudentUpdate, db: Session = Depends(get_db)):
        st = db.get(Student, student_id)
        if not st:
                raise HTTPException(404, 'No encontrado')
        for k, v in payload.dict(exclude_unset=True).items():
                setattr(st, k, v)
        db.commit()
        db.refresh(st)
        return st


@app.delete('/students/{student_id}')
def delete_student(student_id: int, db: Session = Depends(get_db)):
        st = db.get(Student, student_id)
        if not st:
                raise HTTPException(404, 'No encontrado')
        
        # Eliminar archivos físicos del estudiante
        base_dir = os.path.dirname(__file__)
        
        # Eliminar carpeta del estudiante en db/
        db_folder = os.path.abspath(os.path.join(base_dir, '..', 'db', st.code))
        if os.path.exists(db_folder):
                try:
                        shutil.rmtree(db_folder)
                        print(f"[delete] Eliminada carpeta db: {db_folder}")
                except Exception as e:
                        print(f"[delete] Error al eliminar carpeta db {db_folder}: {e}")
        
        # Eliminar archivo de foto en students/
        students_folder = os.path.abspath(os.path.join(base_dir, '..', 'students'))
        # Buscar archivo con cualquier extensión común de imagen
        possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        student_photo_deleted = False
        
        for ext in possible_extensions:
                student_photo = os.path.join(students_folder, f"{st.code}{ext}")
                if os.path.exists(student_photo):
                        try:
                                os.remove(student_photo)
                                print(f"[delete] Eliminada foto: {student_photo}")
                                student_photo_deleted = True
                                break  # Solo eliminar la primera que encuentre
                        except Exception as e:
                                print(f"[delete] Error al eliminar foto {student_photo}: {e}")
        
        if not student_photo_deleted:
                print(f"[delete] No se encontró foto para el estudiante {st.code} en {students_folder}")
        
        db.delete(st)
        db.commit()
        return {"deleted": True, "files_cleaned": True}


# Crear estudiante desde form-data (con foto subida)
@app.post('/students/form', response_model=StudentOut)
async def create_student_form(
        name: str = Form(...),
        grade: Optional[str] = Form(None),
        section: Optional[str] = Form(None),
        gender: Optional[str] = Form(None),
        registration_date: Optional[str] = Form(None),
        code: Optional[str] = Form(None),
        photo: Optional[UploadFile] = File(None),
        db: Session = Depends(get_db)
):
        code = (code or name or f"alumno_{int(datetime.now().timestamp())}").strip().lower().replace(' ', '_')
        existing = db.query(Student).filter_by(code=code).one_or_none()
        if existing:
                raise HTTPException(400, detail='Código ya existe')

        photo_path = None
        if photo and photo.filename:
                base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'students'))
                _ensure_dir(base)
                ext = os.path.splitext(photo.filename)[1] or '.jpg'
                dest = os.path.join(base, f"{code}{ext}")
                with open(dest, 'wb') as f:
                        shutil.copyfileobj(photo.file, f)
                # servirla desde ruta relativa simple
                photo_path = f"/media/students/{os.path.basename(dest)}"
                # copiar a la base de DeepFace para reconocimiento
                faces_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', code))
                _ensure_dir(faces_root)
                faces_dest = os.path.join(faces_root, f"profile{ext}")
                try:
                        shutil.copyfile(dest, faces_dest)
                except Exception:
                        pass

        reg_date = None
        if registration_date:
                try:
                        reg_date = datetime.strptime(registration_date, '%Y-%m-%d').date()
                except Exception:
                        pass

        st = Student(code=code, name=name, grade=grade, section=section, gender=gender, registration_date=reg_date, photo_path=photo_path)
        db.add(st)
        db.commit()
        db.refresh(st)
        return st


# Actualizar estudiante con form-data (y foto opcional)
@app.put('/students/{student_id}/form', response_model=StudentOut)
async def update_student_form(
        student_id: int,
        name: Optional[str] = Form(None),
        grade: Optional[str] = Form(None),
        section: Optional[str] = Form(None),
        gender: Optional[str] = Form(None),
        registration_date: Optional[str] = Form(None),
        photo: Optional[UploadFile] = File(None),
        db: Session = Depends(get_db)
):
        st = db.get(Student, student_id)
        if not st:
                raise HTTPException(404, 'No encontrado')

        if name is not None: st.name = name
        if grade is not None: st.grade = grade
        if section is not None: st.section = section
        if gender is not None: st.gender = gender
        if registration_date:
                try:
                        st.registration_date = datetime.strptime(registration_date, '%Y-%m-%d').date()
                except Exception:
                        pass

        if photo and photo.filename:
                base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'students'))
                _ensure_dir(base)
                ext = os.path.splitext(photo.filename)[1] or '.jpg'
                dest = os.path.join(base, f"{st.code}{ext}")
                with open(dest, 'wb') as f:
                        shutil.copyfileobj(photo.file, f)
                st.photo_path = f"/media/students/{os.path.basename(dest)}"
                # Refrescar copia en DeepFace db
                faces_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', st.code))
                _ensure_dir(faces_root)
                faces_dest = os.path.join(faces_root, f"profile{ext}")
                try:
                        shutil.copyfile(dest, faces_dest)
                except Exception:
                        pass

        db.commit();
        db.refresh(st)
        return st;


# Historial de asistencia por estudiante
@app.get('/students/{student_id}/attendance')
def student_attendance_history(student_id: int = Path(..., ge=1), limit: int = Query(30, ge=1, le=500), db: Session = Depends(get_db)):
        st = db.get(Student, student_id)
        if not st:
                raise HTTPException(404, 'No encontrado')
        rows = (
                db.query(Attendance)
                .filter(Attendance.student_id == student_id)
                .order_by(Attendance.time.desc())
                .limit(limit)
                .all()
        )
        return [
                {
                        'date': str(r.date),
                        'time': r.time.strftime('%H:%M:%S'),
                        'status': r.status,
                } for r in rows
        ]


@app.post('/admin/students/clear-all')
@app.get('/admin/students/clear-all')
def admin_clear_students_all(db: Session = Depends(get_db)):
    # Obtener lista de estudiantes antes de eliminar
    students = db.query(Student).all()
    
    # Eliminar archivos físicos de todos los estudiantes
    base_dir = os.path.dirname(__file__)
    db_base = os.path.abspath(os.path.join(base_dir, '..', 'db'))
    students_base = os.path.abspath(os.path.join(base_dir, '..', 'students'))
    
    files_cleaned = 0
    for st in students:
        # Eliminar carpeta en db/
        db_folder = os.path.join(db_base, st.code)
        if os.path.exists(db_folder):
            try:
                shutil.rmtree(db_folder)
                files_cleaned += 1
                print(f"[clear-all] Eliminada carpeta db: {db_folder}")
            except Exception as e:
                print(f"[clear-all] Error al eliminar carpeta db {db_folder}: {e}")
        
        # Eliminar archivo de foto en students/
        # Buscar archivo con cualquier extensión común de imagen
        possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        
        for ext in possible_extensions:
            student_photo = os.path.join(students_base, f"{st.code}{ext}")
            if os.path.exists(student_photo):
                try:
                    os.remove(student_photo)
                    files_cleaned += 1
                    print(f"[clear-all] Eliminada foto: {student_photo}")
                    break  # Solo eliminar la primera que encuentre
                except Exception as e:
                    print(f"[clear-all] Error al eliminar foto {student_photo}: {e}")
    
    deleted = db.query(Student).delete(synchronize_session=False)
    db.commit()
    return {"deleted": deleted, "files_cleaned": files_cleaned}


# Clase auxiliar para registros de estudiantes ausentes
class AbsentAttendance:
    def __init__(self, student, target_date):
        self.student = student
        self.time = datetime.combine(target_date, datetime.min.time()) if isinstance(target_date, date) else target_date
        self.status = 'Ausente'
        self.confidence = 0.0


@app.get('/export/pdf')
def export_attendance_pdf(
    date: str = Query(..., description="Fecha en formato YYYY-MM-DD"),
    grade: Optional[str] = Query(None, description="Grado a filtrar (1-6)"),
    section: Optional[str] = Query(None, description="Sección a filtrar (A, B, C)"),
    status: Optional[str] = Query(None, description="Estado a filtrar (Puntual, Tarde, Ausente)"),
    db: Session = Depends(get_db)
):
    """Generar reporte de asistencia en formato PDF"""
    try:
        # Validar fecha
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Formato de fecha inválido. Use YYYY-MM-DD")
        
        # Construir consulta base para registros de asistencia
        query = db.query(Attendance).join(Student).filter(
            func.date(Attendance.time) == target_date
        )
        
        # Aplicar filtros opcionales
        if grade and grade.isdigit():
            query = query.filter(Student.grade == int(grade))
        if section:
            query = query.filter(Student.section == section.upper())
        if status and status != 'Ausente':
            query = query.filter(Attendance.status == status)
        
        # Obtener registros de asistencia
        attendances = query.all()
        
        # Si se filtra específicamente por "Ausente", solo mostrar ausentes
        if status == 'Ausente':
            print(f"[PDF DEBUG] Generando reporte solo de ausentes para fecha: {target_date}")
            
            # Obtener todos los estudiantes que deberían haber asistido
            students_query = db.query(Student)
            if grade and grade.isdigit():
                students_query = students_query.filter(Student.grade == int(grade))
            if section:
                students_query = students_query.filter(Student.section == section.upper())
            
            all_students = students_query.all()
            
            # Estudiantes que sí asistieron (aplicando los mismos filtros)
            attended_query = db.query(Attendance).join(Student).filter(
                func.date(Attendance.time) == target_date
            )
            if grade and grade.isdigit():
                attended_query = attended_query.filter(Student.grade == int(grade))
            if section:
                attended_query = attended_query.filter(Student.section == section.upper())
            
            attended_student_ids = {att.student_id for att in attended_query.all()}
            
            # Crear registros ficticios para estudiantes ausentes
            absent_attendances = []
            for student in all_students:
                if student.id not in attended_student_ids:
                    absent_attendances.append(AbsentAttendance(student, target_date))
            
            attendances = absent_attendances
            
        # Si no hay filtro de estado específico o es cualquier otro caso, incluir ausentes también
        elif not status or status == '':  # "Todos los estados"
            print(f"[PDF DEBUG] Generando reporte de TODOS los estados (incluyendo ausentes)")
            
            # Obtener todos los estudiantes que deberían haber asistido
            students_query = db.query(Student)
            if grade and grade.isdigit():
                students_query = students_query.filter(Student.grade == int(grade))
            if section:
                students_query = students_query.filter(Student.section == section.upper())
            
            all_students = students_query.all()
            
            # Estudiantes que sí asistieron (para saber cuáles están ausentes)
            attended_query = db.query(Attendance).join(Student).filter(
                func.date(Attendance.time) == target_date
            )
            if grade and grade.isdigit():
                attended_query = attended_query.filter(Student.grade == int(grade))
            if section:
                attended_query = attended_query.filter(Student.section == section.upper())
            
            attended_student_ids = {att.student_id for att in attended_query.all()}
            
            # Agregar estudiantes ausentes a la lista existente
            for student in all_students:
                if student.id not in attended_student_ids:
                    attendances.append(AbsentAttendance(student, target_date))
            
            print(f"[PDF DEBUG] Total registros (presente + ausentes): {len(attendances)}")
        
        # Crear PDF en memoria
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        
        # Estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.navy,
            alignment=TA_CENTER
        )
        
        # Título del reporte
        title = f"Reporte de Asistencia - {target_date.strftime('%d/%m/%Y')}"
        elements.append(Paragraph(title, title_style))
        
        # Información de filtros aplicados
        filter_info = []
        if grade:
            filter_info.append(f"Grado: {grade}")
        if section:
            filter_info.append(f"Sección: {section}")
        if status:
            filter_info.append(f"Estado: {status}")
        
        if filter_info:
            filter_text = "Filtros aplicados: " + " | ".join(filter_info)
            elements.append(Paragraph(filter_text, styles['Normal']))
            elements.append(Spacer(1, 12))
        
        # Resumen estadístico
        total_records = len(attendances)
        if total_records > 0:
            status_counts = {}
            for att in attendances:
                status_key = att.status
                status_counts[status_key] = status_counts.get(status_key, 0) + 1
            
            summary_text = f"Total de registros: {total_records}<br/>"
            for status_key, count in status_counts.items():
                percentage = (count / total_records) * 100
                summary_text += f"{status_key}: {count} ({percentage:.1f}%)<br/>"
            
            elements.append(Paragraph(summary_text, styles['Normal']))
            elements.append(Spacer(1, 20))
        
        if total_records == 0:
            elements.append(Paragraph("No se encontraron registros para los criterios especificados.", styles['Normal']))
        else:
            # Tabla de datos
            data = [['N°', 'Código', 'Nombre', 'Grado', 'Sección', 'Estado', 'Hora']]
            
            for i, att in enumerate(attendances, 1):
                student = att.student
                time_str = att.time.strftime('%H:%M') if hasattr(att.time, 'strftime') and hasattr(att.time, 'hour') else '--:--'
                
                data.append([
                    str(i),
                    student.code,
                    student.name,
                    str(student.grade),
                    student.section,
                    att.status,
                    time_str
                ])
            
            # Crear tabla
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(table)
        
        # Pie de página con información adicional
        elements.append(Spacer(1, 30))
        footer_text = f"Generado el: {datetime.now().strftime('%d/%m/%Y a las %H:%M')}<br/>Sistema EDUFACE - Control de Asistencia"
        elements.append(Paragraph(footer_text, styles['Normal']))
        
        # Generar PDF
        doc.build(elements)
        buffer.seek(0)
        
        # Retornar como respuesta de streaming
        return StreamingResponse(
            BytesIO(buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=reporte_asistencia_{date}.pdf"}
        )
        
    except Exception as e:
        print(f"Error generando PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando el reporte PDF: {str(e)}")

