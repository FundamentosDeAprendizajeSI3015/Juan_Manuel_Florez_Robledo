#!/usr/bin/env bash
# setup_env.sh — Crea y configura el entorno virtual para FIRE-UdeA
set -e

PYTHON="python3.12"
VENV_DIR=".venv"
KERNEL_NAME="fire-udea"

echo "══════════════════════════════════════════"
echo "  FIRE-UdeA — Setup del entorno virtual"
echo "══════════════════════════════════════════"

# Verificar que Python 3.12 está disponible
if ! command -v $PYTHON &> /dev/null; then
    echo "❌ $PYTHON no encontrado. Instalalo primero:"
    echo "   Ubuntu/WSL: sudo apt install python3.12 python3.12-venv"
    echo "   macOS:      brew install python@3.12"
    exit 1
fi

# Crear venv si no existe
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creando entorno virtual en $VENV_DIR ..."
    $PYTHON -m venv $VENV_DIR
else
    echo "✅ $VENV_DIR ya existe, reutilizando."
fi

# Activar
echo "🔌 Activando entorno virtual..."
source $VENV_DIR/bin/activate

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Registrar kernel de Jupyter
echo "🪐 Registrando kernel de Jupyter '$KERNEL_NAME'..."
python -m ipykernel install --user --name $KERNEL_NAME --display-name "FIRE-UdeA"

echo ""
echo "══════════════════════════════════════════"
echo "  ✅ Listo. Para activar el entorno:"
echo "     source .venv/bin/activate"
echo "══════════════════════════════════════════"
