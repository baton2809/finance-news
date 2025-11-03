# Usa una imagen con Python 3.11 o superior
FROM python:3.11-slim

WORKDIR /app

# Copia tu código al contenedor
COPY finance-news/ .

# Actualiza pip y wheel antes de instalar dependencias
RUN python -m pip install --upgrade pip setuptools wheel

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Evita que Python guarde archivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Comando por defecto
ENTRYPOINT ["python", "main.py"]
