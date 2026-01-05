# fkvisualization
cat > README.md << 'EOF'
# Robot Frames Sandbox

Interactive tool to learn robot frame transforms by building a chain of revolute/prismatic joints + fixed links and visualizing transforms in 3D.

Revolute q is stored internally as rad
Prismatic q is stored internally as meters

## Install

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/robot_frames_ui.py

### Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\robot_frames_ui.py
