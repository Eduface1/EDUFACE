# EDUFACE — Web App de Reconocimiento Facial (DeepFace + FastAPI)

Esta app ofrece endpoints para verificación de rostro, reconocimiento (identificación) y análisis demográfico/emocional usando [DeepFace](https://github.com/serengil/deepface).

## Endpoints
- POST /verify: Verifica si dos imágenes pertenecen a la misma persona.
- POST /recognize: Identifica un rostro contra una base de imágenes (db/).
- POST /analyze: Analiza edad, género, emoción y raza.

## Requisitos
- Python 3.8+
- Dependencias: ver `requirements.txt`.

## Uso rápido
1. Crear y activar entorno virtual (opcional).
2. Instalar dependencias.
3. Iniciar servidor.

```powershell
# 1) (Opcional) Crear venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Instalar dependencias
pip install -r requirements.txt

# 3) Ejecutar servidor
uvicorn app.main:app --reload --port 8002
```

## Base de datos de rostros (para /recognize)
Coloca imágenes por persona en `db/<nombre_persona>/*.jpg`. La app construye o usa representaciones faciales automáticamente.

## Pruebas rápidas
```powershell
# Verificar importación de DeepFace
python -c "from deepface import DeepFace; print('DeepFace OK')"
```

## Notas
- Por defecto usa el detector `retinaface` si está disponible; puede requerir instalación de `opencv-python` y `retinaface` (incluido en requirements).
- En CPU puede tardar en descargar pesos la primera vez.
