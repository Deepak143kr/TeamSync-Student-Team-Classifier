# TeamSync — Student Team Compatibility Classifier
## Local Deployment Guide

---

## Project Structure

```
student-classifier/
├── backend/
│   ├── app.py              ← Flask server + full ML pipeline
│   └── requirements.txt    ← Python dependencies
├── frontend/
│   └── templates/
│       └── index.html      ← Beautiful single-page UI
├── start_windows.bat       ← One-click start (Windows)
├── start_mac_linux.sh      ← One-click start (Mac/Linux)
└── README.md
```

---

## Prerequisites

- **Python 3.8 or newer** — download from https://python.org
- **pip** — comes bundled with Python

---

##  How to Run Locally

### Option A — One-Click Start Scripts (Easiest)

**Windows:**
```
Double-click start_windows.bat
```

**Mac / Linux:**
```bash
chmod +x start_mac_linux.sh
./start_mac_linux.sh
```

---

### Option B — Manual Steps (Any OS)

**Step 1: Open a terminal / command prompt**

**Step 2: Navigate into the backend folder**
```bash
cd student-classifier/backend
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
# (or use pip3 on Mac/Linux)
```

**Step 4: Start the Flask server**
```bash
python app.py
# (or use python3 on Mac/Linux)
```

**Step 5: Open your browser**
```
http://localhost:5000
```

That's it! The app will:
1. Train the full ML pipeline (takes ~10–20 seconds)
2. Print model accuracy in the terminal
3. Serve the web UI at http://localhost:5000

---

##  What Happens on Startup

The server automatically:
- Generates a 5,000-student synthetic dataset
- Scales features with StandardScaler
- Finds optimal K using the Elbow Method → runs K-Means (k=5)
- Finds optimal k for KNN via 5-fold cross-validation
- Finds optimal depth for Decision Tree via 5-fold cross-validation
- Trains both models on 70% of data
- Evaluates on the held-out 30% (accuracy & F1)

---

## API Endpoints

| Method | Endpoint        | Description                    |
|--------|-----------------|--------------------------------|
| GET    | `/`             | Serves the web UI              |
| GET    | `/api/model-info` | Returns trained model stats  |
| POST   | `/api/predict`  | Predicts cluster for a student |

**POST `/api/predict` — Request body:**
```json
{
  "Late_Night_Pref": 5,
  "In_Person_Pref": 2,
  "Stress_Level": 4,
  "Target_Grade": 5,
  "Role_Pref": 5,
  "Skill_Diversity": 3,
  "Pacing": 5
}
```

---

##  To Stop the Server

Press `Ctrl + C` in the terminal.

---

##  Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Port 5000 already in use | Change `port=5000` to `port=5001` in `app.py` and visit `http://localhost:5001` |
| Browser shows "Cannot connect" | Make sure `python app.py` is still running in the terminal |
| Mac: "python not found" | Use `python3` instead of `python` |
