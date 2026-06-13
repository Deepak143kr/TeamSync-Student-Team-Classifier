"""
Flask backend for Student Team Compatibility Classifier
Trains the ML pipeline on startup, then serves predictions via REST API.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
import os

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/templates')
CORS(app)

# Global model state
scaler = None
knn    = None
dt     = None
optimal_knn_k  = None
optimal_depth  = None
acc_knn = acc_dt = f1_knn = f1_dt = None
cluster_profiles = None

FEATURES = [
    'Late_Night_Pref', 'In_Person_Pref', 'Stress_Level',
    'Target_Grade', 'Role_Pref', 'Skill_Diversity', 'Pacing'
]

PERSONA_NAMES = {
    0: "Night-Owl Hustler",
    1: "Balanced Socializer",
    2: "Structured Achiever",
    3: "Solo Coder",
    4: "Relaxed Collaborator",
}

PERSONA_DESCRIPTIONS = {
    0: "Ambitious, high-stress, loves late-night grinding. Best paired with equally driven, fast-paced teammates.",
    1: "Social, low-stress, prefers in-person collaboration. Thrives in friendly, balanced team environments.",
    2: "Structured, goal-oriented, values diverse skills. Excels in well-organized teams with clear roles.",
    3: "Independent, remote-first, fast coder. Works best with autonomy and minimal micromanagement.",
    4: "Relaxed, collaborative, slow-and-steady. Ideal for low-pressure, supportive team cultures.",
}

PERSONA_TRAITS = {
    0: ["🌙 Night Owl", "⚡ High Energy", "🎯 Ambitious", "👑 Leader"],
    1: ["🤝 Social", "😌 Low Stress", "🏢 In-Person", "⚖️ Balanced"],
    2: ["📋 Structured", "🏆 Achiever", "🌈 Diverse Skills", "📊 Organized"],
    3: ["💻 Solo Worker", "🚀 Fast Paced", "🏠 Remote", "🔧 Technical"],
    4: ["😊 Relaxed", "🤗 Collaborative", "🐢 Steady Pace", "💡 Creative"],
}


def generate_dataset():
    np.random.seed(42)
    PERSONA_CONFIGS = {
        "Night-Owl Hustler":     dict(n=1000, late_night=5, in_person=2, stress=5, target=5, role=5, skill=3, pacing=5),
        "Balanced Socializer":   dict(n=1000, late_night=1, in_person=5, stress=2, target=3, role=2, skill=2, pacing=2),
        "Structured Achiever":   dict(n=1000, late_night=3, in_person=4, stress=3, target=5, role=3, skill=5, pacing=2),
        "Solo Coder":            dict(n=1000, late_night=4, in_person=1, stress=1, target=2, role=1, skill=2, pacing=5),
        "Relaxed Collaborator":  dict(n=1000, late_night=2, in_person=3, stress=2, target=2, role=2, skill=4, pacing=1),
    }

    def gen(cfg):
        centers = [cfg['late_night'], cfg['in_person'], cfg['stress'],
                   cfg['target'], cfg['role'], cfg['skill'], cfg['pacing']]
        data = {}
        for feat, center in zip(FEATURES, centers):
            raw = np.random.normal(center, 0.9, cfg['n'])
            data[feat] = np.clip(np.round(raw), 1, 5).astype(int)
        return pd.DataFrame(data)

    frames = [gen(cfg) for cfg in PERSONA_CONFIGS.values()]
    df = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def train_pipeline():
    global scaler, knn, dt, optimal_knn_k, optimal_depth
    global acc_knn, acc_dt, f1_knn, f1_dt, cluster_profiles

    print("[ML] Generating dataset...")
    df = generate_dataset()
    X = df[FEATURES].values

    print("[ML] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[ML] K-Means clustering (k=5)...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    y_labels = kmeans.fit_predict(X_scaled)

    print("[ML] Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_labels, test_size=0.3, random_state=42, stratify=y_labels
    )

    print("[ML] Cross-validating KNN...")
    k_values = list(range(1, 30, 2))
    cv_knn = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5).mean() for k in k_values]
    optimal_knn_k = k_values[np.argmax(cv_knn)]

    print("[ML] Cross-validating Decision Tree...")
    depths = list(range(1, 21))
    cv_dt = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42), X_train, y_train, cv=5).mean() for d in depths]
    optimal_depth = depths[np.argmax(cv_dt)]

    print(f"[ML] Training KNN (k={optimal_knn_k}) and DT (depth={optimal_depth})...")
    knn = KNeighborsClassifier(n_neighbors=optimal_knn_k)
    knn.fit(X_train, y_train)

    dt = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
    dt.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score
    y_pred_knn = knn.predict(X_test)
    y_pred_dt  = dt.predict(X_test)

    acc_knn = round(accuracy_score(y_test, y_pred_knn) * 100, 2)
    acc_dt  = round(accuracy_score(y_test, y_pred_dt)  * 100, 2)
    f1_knn  = round(f1_score(y_test, y_pred_knn, average='weighted') * 100, 2)
    f1_dt   = round(f1_score(y_test, y_pred_dt,  average='weighted') * 100, 2)

    df['Cluster'] = y_labels
    cluster_profiles = df.groupby('Cluster')[FEATURES].mean().round(2).to_dict(orient='index')

    print(f"[ML] Done! KNN Acc={acc_knn}%, DT Acc={acc_dt}%")




@app.route('/')
def index():
    return send_from_directory('../frontend/templates', 'index.html')

@app.route('/api/model-info')
def model_info():
    return jsonify({
        'knn': {'k': optimal_knn_k, 'accuracy': acc_knn, 'f1': f1_knn},
        'dt':  {'depth': optimal_depth, 'accuracy': acc_dt,  'f1': f1_dt},
        'features': FEATURES,
        'cluster_profiles': cluster_profiles,
        'persona_names': PERSONA_NAMES,
        'persona_traits': PERSONA_TRAITS,
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        values = [int(data[f]) for f in FEATURES]
        X_new = np.array([values])
        X_scaled = scaler.transform(X_new)

        pred_knn = int(knn.predict(X_scaled)[0])
        pred_dt  = int(dt.predict(X_scaled)[0])

        knn_votes   = knn.predict_proba(X_scaled)[0]
        dt_proba    = dt.predict_proba(X_scaled)[0]

        return jsonify({
            'knn': {
                'cluster': pred_knn,
                'name': PERSONA_NAMES[pred_knn],
                'description': PERSONA_DESCRIPTIONS[pred_knn],
                'traits': PERSONA_TRAITS[pred_knn],
                'confidence': round(float(knn_votes[pred_knn]) * 100, 1),
                'probabilities': {str(i): round(float(p)*100, 1) for i, p in enumerate(knn_votes)},
            },
            'dt': {
                'cluster': pred_dt,
                'name': PERSONA_NAMES[pred_dt],
                'description': PERSONA_DESCRIPTIONS[pred_dt],
                'traits': PERSONA_TRAITS[pred_dt],
                'confidence': round(float(dt_proba[pred_dt]) * 100, 1),
                'probabilities': {str(i): round(float(p)*100, 1) for i, p in enumerate(dt_proba)},
            },
            'input': dict(zip(FEATURES, values)),
            'agree': pred_knn == pred_dt,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("\nStudent Team Compatibility Classifier\n")
    train_pipeline()
    print("\n🚀 Starting server at http://localhost:5000\n")
    app.run(debug=False, port=5000)
