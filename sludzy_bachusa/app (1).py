from nicegui import ui, app
import pandas as pd
import pickle
from pathlib import Path
import shap
import numpy as np
import numbers
import json

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
TRAIN_FEATURE_ORDER = None  # opcjonalnie wczytana kolejność kolumn z treningu

# --- wczytaj model ---
try:
    with open(BASE_DIR / 'xgboost_best.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
except FileNotFoundError:
    MODELS_LOADED = False
    print("BŁĄD: Nie znaleziono pliku modelu: xgboost_best.pkl")

# --- wczytaj encoder ---
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

# --- wczytaj opcjonalny porządek kolumn z treningu ---
if MODELS_LOADED:
    order_path = BASE_DIR / 'feature_order.json'
    if order_path.exists():
        try:
            TRAIN_FEATURE_ORDER = json.loads(order_path.read_text())
            if not isinstance(TRAIN_FEATURE_ORDER, list) or not TRAIN_FEATURE_ORDER:
                TRAIN_FEATURE_ORDER = None
        except Exception as e:
            print(f"UWAGA: Nie udało się odczytać feature_order.json: {e}")
            TRAIN_FEATURE_ORDER = None

# --- skonfiguruj SHAP ---
if MODELS_LOADED:
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"UWAGA: Nie udało się utworzyć TreeExplainer: {e}")
        explainer = None

# =========================
# 2) DEFINICJE POL / POMOCNICZE
# =========================
FEATURE_CONFIG = {
    "fixed acidity": {"label": "Kwasowość stała", "min": 3.8, "max": 16.0, "step": 0.01},
    "volatile acidity": {"label": "Lotna kwasowość", "min": 0.0, "max": 1.6, "step": 0.01},
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

def decimals_from_step(step):
    s = ('%f' % step).rstrip('0')
    if '.' in s:
        return len(s.split('.')[1])
    return 0

def _normalize_value_for_float(val):
    if isinstance(val, str):
        return val.replace(',', '.')
    return val

def _is_classifier(m):
    return hasattr(m, 'predict_proba')

def decode_class_index(idx: int):
    """0..N -> oryginalna etykieta (np. 3..9) przy użyciu LabelEncodera."""
    arr = np.array([int(idx)], dtype=int)
    decoded = label_encoder.inverse_transform(arr)[0]
    try:
        return int(decoded)  # klasy mogą być stringami '3','4',...
    except Exception:
        return decoded

def _compute_shap(explainer, X_row_df, class_idx):
    """
    Zwraca (expected_value, shap_vals_for_display)
    Obsługuje listę per-klasa i pojedyncze macierze zwracane przez różne wersje SHAP.
    """
    sv = explainer.shap_values(X_row_df)
    ev = explainer.expected_value

    # expected value
    if isinstance(ev, (list, np.ndarray)) and class_idx is not None:
        try:
            expected = float(ev[class_idx])
        except Exception:
            expected = float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)
    else:
        expected = float(ev) if np.isscalar(ev) else float(np.array(ev).ravel()[0])

    # shap values
    if isinstance(sv, list):
        # lista (n_classes) x (n_samples, n_features)
        if class_idx is None:
            sv_arr = np.array(sv[0])[0]
        else:
            sv_arr = np.array(sv[class_idx])[0]
    else:
        sv_np = np.array(sv)
        if sv_np.ndim == 2:
            sv_arr = sv_np[0]
        elif sv_np.ndim == 1:
            sv_arr = sv_np
        else:
            sv_arr = sv_np.reshape(-1)

    return expected, sv_arr

# =========================
# 3) UI I LOGIKA
# =========================
@ui.page('/')
def main_page():
    if not MODELS_LOADED:
        ui.label('❌ Błąd krytyczny: Nie można załadować modeli lub encodera.').classes(
            'text-red-500 text-h5 m-8'
        )
        return

    inputs = {}
    invalid_fields = set()

    def validate_input(feat, val):
        if feat == 'type':
            return val in FEATURE_CONFIG['type']['options']
        if val is None or val == "":
            return False
        try:
            v = float(_normalize_value_for_float(val))
        except (TypeError, ValueError):
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
        empty_fields = []
        other_errors = []
        row_vals = {}

        # 1) Zbierz i zweryfikuj wejścia
        for feat, cfg in FEATURE_CONFIG.items():
            val = inputs[feat].value
            if feat == 'type':
                if val == 'czerwone':
                    row_vals[feat] = 1
                    mark_valid(feat)
                elif val == 'białe':
                    row_vals[feat] = 0
                    mark_valid(feat)
                else:
                    empty_fields.append(cfg["label"])
                    mark_invalid(feat)
            else:
                if val is None or val == "":
                    empty_fields.append(cfg["label"])
                    mark_invalid(feat)
                else:
                    try:
                        v = float(_normalize_value_for_float(val))
                    except (TypeError, ValueError):
                        other_errors.append(f'{cfg["label"]}: niepoprawna wartość')
                        mark_invalid(feat)
                        continue
                    if not (cfg['min'] <= v <= cfg['max']):
                        other_errors.append(f'{cfg["label"]}: poza zakresem ({cfg["min"]}–{cfg["max"]})')
                        mark_invalid(feat)
                    else:
                        row_vals[feat] = v
                        mark_valid(feat)

        if empty_fields or other_errors:
            parts = []
            if empty_fields:
                parts.append("Uzupełnij pola: " + ", ".join(empty_fields))
            if other_errors:
                parts.append("Błędy wartości: " + "; ".join(other_errors))
            msg = "❌ " + " | ".join(parts)
            result_label.text = msg
            result_label.classes(replace='text-red-500 text-lg')
            return

        try:
            # 2) Przygotuj DataFrame w kolejności UI
            cols_ui = list(FEATURE_CONFIG.keys())
            data_df = pd.DataFrame([[row_vals[c] for c in cols_ui]], columns=cols_ui)

            # 2b) (opcjonalnie) wymuś kolejność z treningu
            if TRAIN_FEATURE_ORDER:
                for c in TRAIN_FEATURE_ORDER:
                    if c not in data_df.columns:
                        data_df[c] = 0
                data_df = data_df.reindex(columns=TRAIN_FEATURE_ORDER)

            # 3) Predykcja -> indeks klasy
            y = model.predict(data_df)
            if isinstance(y, np.ndarray) and getattr(y, 'ndim', 1) > 1:
                class_idx = int(np.argmax(y[0]))
            else:
                class_idx = int(y[0])

            # 4) Dekodowanie indeksu -> oryginalna ocena (3..9)
            y_display = decode_class_index(class_idx)

            # 5) Wyświetlenie
            is_good = float(y_display) >= 6.0
            if is_good:
                result_label.text = f'🍷 Ocena wina: {y_display} 😀'
                result_label.classes(replace='text-green-600 text-4xl font-bold')
            else:
                result_label.text = f'🍷 Ocena wina: {y_display} 😡'
                result_label.classes(replace='text-red-600 text-4xl font-bold')

            # 6) SHAP (liczymy dla indeksu klasy)
            if explainer is not None:
                try:
                    feature_names = list(data_df.columns)
                    expected_value, shap_vals = _compute_shap(explainer, data_df, class_idx)
                    force_html = shap.force_plot(
                        expected_value,
                        shap_vals,
                        data_df.values[0],
                        feature_names=feature_names,
                        matplotlib=False,
                        show=False
                    )
                    shap_html_path = BASE_DIR / "static" / "shap_force.html"
                    shap_html_path.parent.mkdir(exist_ok=True)
                    shap.save_html(str(shap_html_path), force_html)

                    # Montowanie (idempotentne)
                    app.add_static_files("/static", str(BASE_DIR / "static"))

                    shap_container.clear()
                    with shap_container:
                        ui.html('<iframe src="/static/shap_force.html" width="100%" height="420" frameborder="0"></iframe>')
                except Exception as e:
                    shap_container.clear()
                    with shap_container:
                        ui.label(f'ℹ️ Nie udało się wygenerować wykresu SHAP: {e}').classes('text-amber-600')
            else:
                shap_container.clear()
                with shap_container:
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
    with ui.card().classes('w-full max-w-4xl mx-auto mt-8 p-8'):
        ui.label('🍇 Predykcja Jakości Wina').classes('text-h4 self-center font-bold mb-4')

        with ui.grid(columns=2).classes('w-full gap-4'):
            for feat, cfg in FEATURE_CONFIG.items():
                if feat == 'type':
                    inputs[feat] = (
                        ui.select(
                            label=cfg['label'],
                            options=cfg['options'],
                            value=None,
                            with_input=False
                        )
                        .props('outlined dense')
                        .classes('w-full')
                        .on('update:model-value', lambda e, f=feat: on_change(f, e))
                    )
                else:
                    inp = (
                        ui.input(label=cfg['label'], value=None)
                        .props('outlined dense')
                        .classes('group w-full')
                    )
                    with inp.add_slot('append'):
                        with ui.column().classes(
                            'items-center opacity-0 group-hover:opacity-100 transition-opacity duration-200'
                        ):
                            ui.button('▲', on_click=lambda _, f=feat, s=cfg['step']: adjust_value(f, s)).props('flat dense round size=sm')
                            ui.button('▼', on_click=lambda _, f=feat, s=cfg['step']: adjust_value(f, -s)).props('flat dense round size=sm')

                    inp.on('keydown', lambda e, f=feat, step=cfg['step']:
                           (adjust_value(f, step) if getattr(e, 'key', '') == 'ArrowUp'
                            else (adjust_value(f, -step) if getattr(e, 'key', '') == 'ArrowDown' else None)))
                    inp.on('blur', lambda e, f=feat: on_change(f, inputs[f].value))
                    inputs[feat] = inp

        with ui.row().classes('w-full justify-end gap-2 mt-6'):
            ui.button('🧹 Wyczyść', on_click=clear_inputs, color='amber')
            ui.button('🔮 Oceń Wino', on_click=handle_prediction, color='green')

        result_label = ui.label('Ocena wina pojawi się tutaj...').classes('text-lg self-center mt-4 text-gray-500')
        shap_container = ui.column().classes('w-full mt-6')

# =========================
# 4) RUN
# =========================
# reload=True jest wygodne lokalnie; na produkcji rozważ False
ui.run(title="Wine Quality Predictor", dark=True, reload=True)

