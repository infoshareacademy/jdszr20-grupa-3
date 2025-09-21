from nicegui import ui
import pandas as pd
import pickle
from pathlib import Path
import shap
import numpy as np
import json
import plotly.graph_objects as go

# =========================
# 1) KONFIGURACJA / ZAŁADUNEK
# =========================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

MODELS_LOADED = True
model = None
label_encoder = None
explainer = None
TRAIN_FEATURE_ORDER = None

# --- model ---
try:
    with open(BASE_DIR / 'xgboost_best.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
except FileNotFoundError:
    MODELS_LOADED = False
    print("BŁĄD: Nie znaleziono pliku modelu: xgboost_best.pkl")

# --- encoder ---
if MODELS_LOADED:
    encoder_path = BASE_DIR / 'encoder_le.pkl'
    try:
        with open(encoder_path, 'rb') as f_encoder:
            label_encoder = pickle.load(f_encoder)
        print('Encoder type:', type(label_encoder))
        if not hasattr(label_encoder, 'classes_'):
            raise RuntimeError('encoder_le.pkl istnieje, ale nie ma atrybutu classes_ (to nie jest LabelEncoder).')
        print('Encoder classes_:', getattr(label_encoder, 'classes_', None))

        if hasattr(model, 'classes_'):
            if len(label_encoder.classes_) != len(model.classes_):
                raise RuntimeError(
                    f'LabelEncoder i model mają różną liczbę klas: '
                    f'{len(label_encoder.classes_)} vs {len(model.classes_)}'
                )
    except FileNotFoundError:
        MODELS_LOADED = False
        print(f"BŁĄD: Nie znaleziono pliku encodera: {encoder_path}")
    except Exception as e:
        MODELS_LOADED = False
        print(f"BŁĄD encodera: {e}")

# --- kolejność kolumn ---
if MODELS_LOADED:
    order_path = BASE_DIR / 'feature_order.json'
    if order_path.exists():
        try:
            TRAIN_FEATURE_ORDER = json.loads(order_path.read_text())
            if not isinstance(TRAIN_FEATURE_ORDER, list) or not TRAIN_FEATURE_ORDER:
                TRAIN_FEATURE_ORDER = None
        except Exception:
            TRAIN_FEATURE_ORDER = None

# --- SHAP explainer ---
if MODELS_LOADED:
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"UWAGA: Nie udało się utworzyć TreeExplainer: {e}")
        explainer = None

# =========================
# 2) KONFIG POLA
# =========================
FEATURE_CONFIG = {
    "fixed acidity": {"label": "Kwasowość stała [g/L]", "min": 3.8, "max": 16.0, "step": 0.01},
    "volatile acidity": {"label": "Lotna kwasowość [g/L]", "min": 0.0, "max": 1.6, "step": 0.01},
    "citric acid": {"label": "Kwas cytrynowy [g/L]", "min": 0.0, "max": 1.7, "step": 0.01},
    "residual sugar": {"label": "Cukier resztkowy [g/L]", "min": 0.6, "max": 65.8, "step": 0.1},
    "chlorides": {"label": "Zawartość chlorków [g/L]", "min": 0.0, "max": 0.7, "step": 0.001},
    "free sulfur dioxide": {"label": "Wolny dwutlenek siarki [mg/L]", "min": 1.0, "max": 290.0, "step": 0.1},
    "total sulfur dioxide": {"label": "Łączna ilość dwutlenku siarki [mg/L]", "min": 6.0, "max": 450.0, "step": 0.1},
    "density": {"label": "Gęstość [g/cm³]", "min": 0.98, "max": 1.04, "step": 0.0001},
    "pH": {"label": "pH", "min": 2.7, "max": 4.2, "step": 0.01},
    "sulphates": {"label": "Zawartość siarczanów [g/L]", "min": 0.2, "max": 2.0, "step": 0.01},
    "alcohol": {"label": "Alkohol [%]", "min": 8.0, "max": 15.0, "step": 0.1},
    "type": {"label": "Rodzaj wina", "options": ["czerwone", "białe"]},
}

FEATURE_LABELS_PL = {feat: cfg['label'] for feat, cfg in FEATURE_CONFIG.items()}


def decimals_from_step(step):
    s = ('%f' % step).rstrip('0')
    if '.' in s:
        return len(s.split('.')[1])
    return 0

def _normalize_value_for_float(val):
    if isinstance(val, str):
        return val.replace(',', '.').strip()
    return val

def _compute_shap(explainer, X_row_df, class_idx):
    sv = explainer.shap_values(X_row_df)
    ev = explainer.expected_value

    if isinstance(ev, (list, np.ndarray)):
        ev_arr = np.array(ev)
        ev_flat = ev_arr.ravel()
        if class_idx is not None and ev_flat.size > 1:
            expected = float(ev_flat[class_idx])
        else:
            expected = float(ev_flat[0])
    else:
        expected = float(np.array(ev).ravel()[0])

    if isinstance(sv, list):
        sv_arr = np.array(sv[class_idx], dtype=float)
        shap_vec = sv_arr[0] if sv_arr.ndim == 2 else sv_arr.ravel()
    else:
        sv_np = np.array(sv, dtype=float)
        if sv_np.ndim == 3:
            shap_vec = sv_np[0, :, class_idx]
        elif sv_np.ndim == 2:
            shap_vec = sv_np[0, :]
        elif sv_np.ndim == 1:
            shap_vec = sv_np
        else:
            raise ValueError(f"Nieoczekiwany kształt shap_values: {sv_np.shape}")

    return expected, shap_vec

# =========================
# 3) UI
# =========================
@ui.page('/')
def main_page():
    if not MODELS_LOADED:
        ui.label('❌ Błąd krytyczny: Nie można załadować modeli').classes('text-red-500 text-h5 m-8')
        return

    inputs = {}
    invalid_fields = set()

    def validate_input(feat, val):
        if feat == 'type':
            return val in FEATURE_CONFIG['type']['options']
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return False
        try:
            v = float(_normalize_value_for_float(val))
        except Exception:
            return False
        cfg = FEATURE_CONFIG[feat]
        return cfg['min'] <= v <= cfg['max']

    def mark_invalid(feat):
        invalid_fields.add(feat)
        try:
            inputs[feat].classes(add='border-red-500')
        except Exception:
            pass

    def mark_valid(feat):
        invalid_fields.discard(feat)
        try:
            inputs[feat].classes(remove='border-red-500')
        except Exception:
            pass

    def on_change(feat, value=None):
        if hasattr(value, 'value'):
            val = value.value
        else:
            val = inputs[feat].value if value is None else value

        if not validate_input(feat, val):
            mark_invalid(feat)
            if val not in (None, ""):
                cfg = FEATURE_CONFIG[feat]
                ui.notify(
                    f'⚠️ {cfg["label"]}: wartość poza zakresem ({cfg["min"]}–{cfg["max"]}) lub niepoprawna',
                    color='negative'
                )
        else:
            mark_valid(feat)

    def handle_prediction():
        row_vals = {}
        empty_fields, other_errors = [], []

        # === Walidacja ===
        for feat, cfg in FEATURE_CONFIG.items():
            val = inputs[feat].value
            if feat == 'type':
                if val == 'czerwone': row_vals[feat] = 1; mark_valid(feat)
                elif val == 'białe': row_vals[feat] = 0; mark_valid(feat)
                else: empty_fields.append(cfg["label"]); mark_invalid(feat)
            else:
                if val in (None, ""):
                    empty_fields.append(cfg["label"])
                    mark_invalid(feat)
                    continue
                try:
                    v = float(_normalize_value_for_float(val))
                except:
                    other_errors.append(f'{cfg["label"]}: niepoprawna wartość')
                    mark_invalid(feat)
                    continue
                if not (cfg['min'] <= v <= cfg['max']):
                    other_errors.append(f'{cfg["label"]}: poza zakresem (dozwolone {cfg["min"]} – {cfg["max"]})')
                    mark_invalid(feat)
                else:
                    row_vals[feat] = v
                    mark_valid(feat)

        if empty_fields or other_errors:
            msg = "❌ " + " | ".join([
                ("Uzupełnij: " + ", ".join(empty_fields)) if empty_fields else "",
                ("Błędy: " + "; ".join(other_errors)) if other_errors else ""
            ])
            result_label.text = msg
            result_label.classes(replace='text-red-500 text-lg')
            return
        

        # === Predykcja ===
        try:
            cols_ui = list(FEATURE_CONFIG.keys())
            data_df = pd.DataFrame([[row_vals[c] for c in cols_ui]], columns=cols_ui)

            print("DEBUG: data_df:")
            print(data_df)
            print("Columns order:", list(data_df.columns))
            print("TRAIN_FEATURE_ORDER:", TRAIN_FEATURE_ORDER)

            if TRAIN_FEATURE_ORDER:
                for c in TRAIN_FEATURE_ORDER:
                    if c not in data_df.columns:
                        data_df[c] = 0
                data_df = data_df.reindex(columns=TRAIN_FEATURE_ORDER)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(data_df)
                class_idx = int(np.argmax(proba[0]))
            else:
                class_idx = int(model.predict(data_df)[0])

            y_display = label_encoder.inverse_transform([class_idx])[0]

            print("Model classes_:", getattr(model, "classes_", None))
            print("Encoder classes_:", getattr(label_encoder, "classes_", None))
            print("Argmax idx:", class_idx, "=> Decoded:", y_display)

            is_good = float(y_display) >= 6.0
            if is_good:
                result_label.text = f'🍷 Ocena wina: {y_display} 🍷😎'
                result_label.classes(replace='text-green-600 text-4xl font-bold')
            else:
                result_label.text = f'🍷 Ocena wina: {y_display} 🤮'
                result_label.classes(replace='text-red-600 text-4xl font-bold')

            shap_container.clear()
            with shap_container:
                if explainer is not None:
                    try:
                        expected_value, shap_vals = _compute_shap(explainer, data_df, class_idx)
                        shap_vals = np.array(shap_vals, dtype=float).ravel()
                        features_vec = data_df.values[0]
                        feature_names = list(data_df.columns)

                        if shap_vals.shape[0] != features_vec.shape[0]:
                            m = min(shap_vals.shape[0], features_vec.shape[0])
                            shap_vals = shap_vals[:m]
                            features_vec = features_vec[:m]
                            feature_names = feature_names[:m]

                        feature_names_pl = [FEATURE_LABELS_PL.get(f, f) for f in feature_names]
                        top_idx = np.argsort(np.abs(shap_vals))[::-1][:12]
                        top_vals = shap_vals[top_idx]
                        top_features = [feature_names_pl[i] for i in top_idx]

                        fig = go.Figure(go.Bar(
                            x=top_vals,
                            y=top_features,
                            orientation='h',
                            text=[f"SHAP: {v:.2f}<br>Value: {features_vec[top_idx[i]]:.2f}" 
                                for i, v in enumerate(top_vals)],
                            hoverinfo='text'
                            ))
                        fig.update_layout(
                            title="Wpływ cech (SHAP) na predykcję",
                            autosize=False,
                            width=1200,
                            height=800,
                            margin=dict(l=150, r=50, t=80, b=50),
                            yaxis=dict(autorange="reversed")
                        )
                        ui.plotly(fig).classes('w-full h-[800px]')
                    except Exception as e:
                        ui.label(f'ℹ️ Nie udało się wygenerować wykresu SHAP: {e}').classes('text-amber-600')
                else:
                    ui.label('ℹ️ SHAP nie został zainicjalizowany.').classes('text-amber-600')

        except Exception as e:
            result_label.text = f'❌ Błąd podczas predykcji: {e}'
            result_label.classes(replace='text-red-500 text-lg')

    def clear_inputs():
        for feat in inputs:
            inputs[feat].set_value(None)
            try:
                inputs[feat].classes(remove='border-red-500')
            except Exception:
                pass
        invalid_fields.clear()
        result_label.text = 'Ocena wina pojawi się tutaj...'
        result_label.classes(replace='text-gray-500 text-lg')
        shap_container.clear()

    def adjust_value(feat, delta):
        cfg = FEATURE_CONFIG[feat]
        step = cfg['step']
        cur = inputs[feat].value
        try:
            v = float(_normalize_value_for_float(cur)) if cur not in (None, "") else cfg['min']
        except ValueError:
            v = cfg['min']
        v_new = v + delta
        v_new = min(cfg['max'], max(cfg['min'], v_new))
        decs = decimals_from_step(step)
        fmt = f"{{:.{decs}f}}"
        inputs[feat].set_value(fmt.format(v_new))
        on_change(feat, fmt.format(v_new))

    # --- Layout ---
    with ui.card().classes('w-full max-w-none mx-auto mt-8 p-8'):
        ui.label('🍇 Predykcja Jakości Wina').classes('text-h4 self-center font-bold mb-4')
        with ui.grid(columns=2).classes('w-full gap-4'):
            for feat, cfg in FEATURE_CONFIG.items():
                if feat == 'type':
                    inputs[feat] = ui.select(label=cfg['label'], options=cfg['options']).props('outlined dense').classes('w-full').on('update:model-value', lambda e, f=feat: on_change(f, e))
                else:
                    inp = ui.input(label=cfg['label']).props('outlined dense').classes('group w-full')
                    with inp.add_slot('append'):
                        with ui.column().classes('items-center opacity-0 group-hover:opacity-100'):
                            ui.button('▲', on_click=lambda _, f=feat, s=cfg['step']: adjust_value(f, s)).props('flat dense round size=sm')
                            ui.button('▼', on_click=lambda _, f=feat, s=cfg['step']: adjust_value(f, -s)).props('flat dense round size=sm')
                    inp.on('blur', lambda e, f=feat: on_change(f, inputs[f].value))
                    inputs[feat] = inp
        with ui.row().classes('w-full justify-end gap-2 mt-6'):
            ui.button('🧹 Wyczyść', on_click=clear_inputs, color='amber')
            ui.button('🔮 Oceń Wino', on_click=handle_prediction, color='green')

        result_label = ui.label('Ocena wina pojawi się tutaj...').classes('text-lg self-center mt-4 text-gray-500')
        shap_container = ui.column().classes('w-full mt-6 px-4')


# =========================
# 4) RUN
# =========================
ui.run(title="Wine Quality Predictor", dark=True, reload=True)
