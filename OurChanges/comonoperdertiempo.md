# Cómo no perder tiempo iterando con Docker en Voicebox

## Regla de oro

**Nunca hagas `docker compose up --build` para probar un cambio de código Python.**
Un rebuild completo tarda ~15 minutos. Hay formas de aplicar cambios en segundos.

---

## Iteración rápida sin rebuild

### 1. Cambios en código Python (`backend/`)

Editar el archivo local, luego copiarlo al contenedor y reiniciar:

```powershell
docker cp backend/backends/pytorch_backend.py voicebox:/app/backend/backends/pytorch_backend.py
docker restart voicebox
```

**Tiempo: ~5 segundos**

### 2. Instalar paquetes del sistema (apt) en el contenedor

```powershell
docker exec voicebox apt-get update -qq
docker exec voicebox apt-get install -y --no-install-recommends sox
```

**Tiempo: ~10 segundos**

### 3. Guardar cambios en la imagen (para que `docker compose up` los use)

Después de hacer `docker cp` y/o `docker exec apt-get install`:

```powershell
docker commit voicebox voicebox-voicebox:latest
```

Ahora `docker compose up` (sin `--build`) usará la imagen actualizada.

**Tiempo: ~2 segundos**

### 4. Levantar el contenedor

```powershell
docker compose up
```

Sin `--build`, usa la imagen cacheada/commiteada.

---

## Flujo completo de un cambio rápido

```powershell
# 1. Editar el archivo local (VS Code)
# 2. Copiar al contenedor
docker cp backend/backends/mi_archivo.py voicebox:/app/backend/backends/mi_archivo.py
# 3. Reiniciar
docker restart voicebox
# 4. Ver logs
docker logs voicebox --tail 20
# 5. Si funciona, guardar en la imagen
docker commit voicebox voicebox-voicebox:latest
```

---

## Cuándo SÍ hacer rebuild

Solo cuando cambias:
- `Dockerfile` (estructura, stages, dependencias)
- `backend/requirements.txt` (nuevas dependencias Python)
- `package.json` o frontend (build de Vite)
- `.dockerignore`

En esos casos no hay alternativa al rebuild. Pero los cambios de código Python
**nunca** requieren rebuild.

---

## Errores conocidos y fixes

### `NameError: name 'force_offline_if_cached' is not defined`

Falta el import en `backend/backends/pytorch_backend.py`:

```python
from ..utils.hf_offline_patch import force_offline_if_cached
```

### `sox: not found`

Instalar en el contenedor:

```powershell
docker exec voicebox apt-get update -qq
docker exec voicebox apt-get install -y --no-install-recommends sox
docker commit voicebox voicebox-voicebox:latest
```

### `exec /usr/local/bin/entrypoint.sh: no such file or directory`

El script `scripts/rocm-entrypoint.sh` tiene line endings Windows (CRLF).
Convertir a LF:

```powershell
(Get-Content scripts/rocm-entrypoint.sh -Raw) -replace "`r`n", "`n" | Set-Content -NoNewline scripts/rocm-entrypoint.sh
```

### GPU no detectada (`cuda available: False`)

Falta la sección `deploy.resources.reservations.devices` en `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Recrear el contenedor (sin rebuild):

```powershell
docker compose up
```

---

## Estado actual de la rama `mio`

- **Dockerfile**: multi-stage con uv, venv, flash-attn wheel, turboquant, bitsandbytes, triton, sox, soporte ROCm
- **docker-compose.yml**: GPU NVIDIA habilitada, env vars de TurboQuant/quantization
- **pytorch_backend.py**: import de `force_offline_if_cached`, quantización bitsandbytes, cache_dir unificado
- **hume_backend.py**: quantización bitsandbytes + TurboQuant
- **turboquant_cache.py**: nuevo módulo de compresión KV cache
- **Imagen Docker**: `voicebox-voicebox:latest` commiteada con sox + import fix

### Commits en `origin/mio`:

```
bf6d9e4 Merge remote-tracking branch 'origin/main' into mio
15b83de feat: add TurboQuant KV cache compression and bitsandbytes quantization
136db82 Align torch with prebuilt flash-attn wheel
```

### Cambios sin commitear (en disco):

- `.dockerignore`: excepción para `scripts/rocm-entrypoint.sh`
- `Dockerfile`: sox añadido, entrypoint reordenado, node fix trailing comma
- `docker-compose.yml`: GPU reservations
- `backend/backends/pytorch_backend.py`: import `force_offline_if_cached`
- `scripts/rocm-entrypoint.sh`: convertido a LF

Estos cambios están en la imagen Docker commiteada pero **no** en git.
Hacer commit cuando se quiera persistir.