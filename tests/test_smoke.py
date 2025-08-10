def test_imports():
    import importlib
    # Siempre requeridos
    for mod in ["fastapi", "uvicorn"]:
        importlib.import_module(mod)
    # DeepFace puede instalarse después; omitir si falta
    try:
        importlib.import_module("deepface")
    except ModuleNotFoundError:
        pass
