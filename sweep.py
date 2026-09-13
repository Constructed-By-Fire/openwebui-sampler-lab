"""Run an independent Open WebUI generation for each sampling value."""

import csv
from datetime import datetime, timezone
from html import escape
import json
import math
import re
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request


# Experiment configuration
MODEL = "overlord"
PROMPT = "I bought you a ridiculous little hat."
PRESETS = {
    "temperature": [0.1, 0.4, 0.7, 1.0, 1.3, 1.5],
    "top_p": [0.25, 0.45, 0.65, 0.8, 0.9, 1.0],
    "top_k": [5, 10, 20, 40, 100, 0],
    "min_p": [0, 0.01, 0.03, 0.07, 0.12, 0.2],
    "repeat_penalty": [0.95, 1.0, 1.08, 1.15, 1.22, 1.3],
}
INTERACTION_RANGES = {
    "temperature": [0.3, 0.6, 0.9, 1.2, 1.5],
    "top_p": [0.3, 0.5, 0.7, 0.85, 1.0],
    "top_k": [5, 15, 40, 100, 0],
    "min_p": [0, 0.03, 0.07, 0.12, 0.2],
    "repeat_penalty": [0.95, 1.0, 1.1, 1.2, 1.3],
}
# Lower, middle, upper, and fine diagnostic reference ranges (not model baselines).
RANGE_PRESETS = {
    "temperature": [[0.1, 0.2, 0.3, 0.4], [0.4, 0.7, 0.9, 1.0], [1.0, 1.2, 1.3, 1.5], [0.7, 0.75, 0.8, 0.85, 0.9]],
    "top_p": [[0.25, 0.35, 0.45, 0.55], [0.45, 0.65, 0.75, 0.8], [0.8, 0.9, 0.95, 1.0], [0.75, 0.775, 0.8, 0.825, 0.85]],
    "top_k": [[5, 8, 10, 15], [15, 20, 40, 60], [60, 80, 100, 0], [25, 30, 40, 50, 55]],
    "min_p": [[0, 0.01, 0.02, 0.03], [0.03, 0.05, 0.07, 0.1], [0.1, 0.12, 0.16, 0.2], [0.05, 0.06, 0.07, 0.08, 0.09]],
    "repeat_penalty": [[0.95, 0.98, 1.0, 1.04], [1.0, 1.08, 1.12, 1.15], [1.15, 1.2, 1.25, 1.3], [1.08, 1.1, 1.12, 1.14, 1.16]],
}
MATRIX_SEEDS = [1, 2, 3, 4, 5]
# Supported CSV/advanced experiment fields, not default request values.
PARAMETERS = (
    "seed", "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "num_ctx", "num_predict",
)

TIMEOUT_SECONDS = 180
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIELDS = [
    "timestamp", "model", "prompt", "sweep_parameter", "sweep_value",
    *PARAMETERS, "response", "error", "preset_baseline", "experiment_overrides",
]


def read_config():
    """Read simple KEY=value lines; optionally quoted values are supported."""
    config = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator:
            raise ValueError("Invalid .env line; expected KEY=value.")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        config[name.strip()] = value
    url = config.get("OPENWEBUI_URL", "").rstrip("/")
    key = config.get("OPENWEBUI_API_KEY", "")
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Set OPENWEBUI_URL to an HTTP(S) base URL without credentials.")
    if not key or key == "your-key-here" or "\n" in key or "\r" in key:
        raise ValueError("Set OPENWEBUI_API_KEY in the local .env file.")
    return url, key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward the authentication header to a redirected destination.
        return None


def generate(opener, url, key, settings, prompt, model=MODEL):
    # Open WebUI 0.11.3 forwards this options object to its Ollama backend.
    # Construct a fresh message list every time; the preset supplies the persona.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": settings.copy(),
        "stream": False,
    }
    request = urllib.request.Request(
        url + "/api/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as reply:
            if "text/event-stream" in reply.headers.get("Content-Type", ""):
                return "", "Server enabled streaming despite stream=false; disable it in the preset."
            data = json.load(reply)
        if not isinstance(data, dict) or data.get("error"):
            return "", "API returned an error or unexpected response."
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            return "", "API returned no text response."
        return content, ""
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return "", f"HTTP {exc.code}: authentication or API access denied."
        return "", f"HTTP {exc.code}: generation request failed."
    except (TimeoutError, socket.timeout):
        return "", f"Request timed out after {TIMEOUT_SECONDS} seconds."
    except urllib.error.URLError:
        return "", "Unable to reach Open WebUI; check its URL and service."
    except (ValueError, KeyError, IndexError, TypeError):
        return "", "API returned invalid JSON or an unexpected response structure."
    except Exception:
        # Never expose raw exceptions or server error bodies containing secrets.
        return "", "Generation failed while sending or reading the request."


# Shared palette and typography for every report; layout rules stay separate.
THEME_CSS = """
:root{color-scheme:light dark;--page:#f5f5f5;--surface:white;--border:#ccc;
 --text:#222;--secondary:#555;--error:#a11919;--table-header:#ededed}
@media (prefers-color-scheme: dark){
 :root{--page:#0b0b0b;--surface:#171717;--border:#2a2a2a;
 --text:#e8e8e8;--secondary:#b8b8b8;--error:#ff9c9c;--table-header:#202020}}
body{font:17px/1.6 system-ui,sans-serif;background:var(--page);color:var(--text)}
h2{margin-top:0}
article,.matrix th,.matrix td{background:var(--surface);border-color:var(--border)}
.matrix th{background:var(--table-header)}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;color:var(--text)}
small{color:var(--secondary)}
.error{color:var(--error)}
"""

CARD_CSS = """
body{max-width:900px;margin:40px auto;padding:0 20px}
article{border:1px solid var(--border);border-radius:8px;padding:20px;margin:24px 0}
"""

MATRIX_CSS = """
body{max-width:none;margin:40px 0;padding:0 32px}
body>h1,body>p{max-width:900px}
.matrix-scroll{overflow:auto;max-height:80vh;margin:24px 0;isolation:isolate}
.matrix{border-collapse:separate;border-spacing:0;width:100%;table-layout:fixed}
.matrix th,.matrix td{padding:16px;border-right:1px solid var(--border);
 border-bottom:1px solid var(--border);vertical-align:top;text-align:left;
 box-sizing:border-box;overflow-wrap:anywhere}
.matrix col.seed-column{width:100px}
.matrix thead th{position:sticky;top:0;z-index:2;border-top:1px solid var(--border)}
.matrix th:first-child{position:sticky;left:0;z-index:1;border-left:1px solid var(--border)}
.matrix thead th:first-child{z-index:3}
.matrix pre{margin-bottom:0}
"""


def report_css(is_matrix):
    return THEME_CSS + (MATRIX_CSS if is_matrix else CARD_CSS)


def matrix_html(rows, parameter, row_parameter="seed"):
    """Render saved responses without changing experiment or CSV ordering."""
    safe = lambda value: escape(str(value), quote=True)
    values = list(dict.fromkeys(row[parameter] for row in rows))
    row_values = list(dict.fromkeys(row[row_parameter] for row in rows))
    cells = {}
    for row in rows:
        cells.setdefault((row[row_parameter], row[parameter]), []).append(row)
    row_label = 'Seed' if row_parameter == 'seed' else row_parameter
    corner = 'Seed' if row_parameter == 'seed' else f'{row_parameter} ↓ / {parameter} →'
    parts = ['<div class="matrix-scroll" role="region" aria-label="Experiment comparison" tabindex="0">',
             f'<table class="matrix" style="min-width:{100 + 260 * len(values)}px">'
             '<colgroup><col class="seed-column">'
             + '<col>' * len(values)
             + f'</colgroup><thead><tr><th scope="col">{safe(corner)}</th>']
    parts.extend(f'<th scope="col">{safe(parameter)} {safe(format_value(parameter, value))}</th>' for value in values)
    parts.append('</tr></thead><tbody>')
    for row_value in row_values:
        parts.append(f'<tr><th scope="row">{safe(row_label)} {safe(format_value(row_parameter, row_value))}</th>')
        for value in values:
            parts.append('<td>')
            for row in cells.get((row_value, value), []):
                parts.append(f'<small>{safe(row["timestamp"])}</small>')
                if row["error"]:
                    parts.append(f'<p class="error">{safe(row["error"])}</p>')
                parts.append(f'<pre>{safe(row["response"])}</pre>')
            parts.append('</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    return ''.join(parts)


def result_basename(parameter, seeds, fixed_seed, y_parameter, stamp):
    def safe_name(name):
        return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "parameter"

    x = safe_name(parameter)
    if y_parameter is not None:
        prefix = f"{safe_name(y_parameter)}_x_{x}_seed{int(fixed_seed)}"
    elif seeds is not None:
        prefix = f"{x}_multiseed"
    else:
        prefix = f"{x}_quick_seed{int(fixed_seed)}"
    # Keep microseconds for same-second runs; leave the report's UTC stamp unchanged.
    timestamp = datetime.strptime(stamp, "%Y%m%dT%H%M%S_%fZ").strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}"


def write_reports(rows, stamp, parameter, values, seeds, prompt, fixed_seed=1, y_parameter=None, y_values=None, model=MODEL, baseline=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    basename = result_basename(parameter, seeds, fixed_seed, y_parameter, stamp)
    csv_path = RESULTS / f"{basename}.csv"
    html_path = RESULTS / f"{basename}.html"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    safe = lambda value: escape(str(value), quote=True)
    overrides = {parameter: values, "seed": seeds if seeds is not None else [fixed_seed]}
    if y_parameter:
        overrides[y_parameter] = y_values
    is_matrix = seeds is not None or y_parameter is not None
    heading = f"Two-parameter grid — fixed seed {fixed_seed}" if y_parameter else "Open WebUI sampling experiment"
    cards = []
    for row in rows:
        label = f'{parameter} = {row["sweep_value"]}'
        error = f'<p class="error">{safe(row["error"])}</p>' if row["error"] else ""
        cards.append(
            f'<article><h2>{safe(label)}</h2>'
            f'<small>{safe(row["timestamp"])}</small>{error}'
            f'<pre>{safe(row["response"])}</pre></article>'
        )
    content = matrix_html(rows, parameter, y_parameter or "seed") if is_matrix else "".join(cards)
    html_path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Open WebUI sampling experiment</title><style>'
        + report_css(is_matrix) + '</style>'
        f'<h1>{safe(heading)}</h1>'
        f'<p>Model: <strong>{safe(model)}</strong> · UTC run: {safe(stamp)}</p>'
        f'<p>Prompt: {safe(prompt)}</p>'
        f'<p>Sweep / X: {safe(parameter)} = {safe(values)}</p>'
        + (f'<p>Y: {safe(y_parameter)} = {safe(y_values)}</p>' if y_parameter else '') +
        f'<p>Seeds: {safe(seeds if seeds is not None else [fixed_seed])} · Generations: {len(rows)}</p>'
        f'<p>Preset baseline: {safe(json.dumps(baseline or {}))}</p>'
        f'<p>Experiment overrides: {safe(json.dumps(overrides))}</p>'
        '<p>Backend defaults: parameters absent from the preset and experiment are not explicitly set. '
        'Their effective values are supplied by Open WebUI/Ollama. Saved values for tested parameters '
        'are superseded by the experiment overrides.</p>'
        '<p>Each request uses only the user prompt. The saved preset supplies the persona. '
        'Settings shown are requested values; a fixed seed does not guarantee determinism.</p>'
        + content + '</html>', encoding="utf-8",
    )
    return csv_path, html_path


def experiment_settings(parameter, values, seeds, fixed_seed=1, y_parameter=None, y_values=None):
    # Send only experimental overrides. Open WebUI applies the selected preset baseline.
    for value in values:
        for row_value in (y_values if y_parameter else (seeds if seeds is not None else [fixed_seed])):
            settings = {"seed": fixed_seed, parameter: value}
            settings[y_parameter or "seed"] = row_value
            yield settings


def parse_values(text, parameter):
    integers = {"seed", "top_k", "num_ctx", "num_predict"}
    values = []
    for part in text.split(","):
        value = int(part.strip()) if parameter in integers else float(part.strip())
        if not math.isfinite(value):
            raise ValueError("Values must be finite numbers.")
        if parameter in {"top_p", "min_p"} and not 0 <= value <= 1:
            raise ValueError("Values must be between 0 and 1.")
        if parameter in {"num_ctx", "num_predict"} and value < 1:
            raise ValueError("Values must be positive integers.")
        if parameter not in {"presence_penalty", "frequency_penalty"} and value < 0:
            raise ValueError("Values must be nonnegative.")
        values.append(value)
    return values


def ask_values(parameter):
    while True:
        text = input(f"{parameter} comma-separated values (Enter to go back): ").strip()
        if not text:
            return None
        try:
            return parse_values(text, parameter)
        except (ValueError, OverflowError):
            print("Invalid values. Use finite numbers, integers for seed/top_k/token counts, "
                  "0–1 for top_p/min_p, and positive token counts.")


def format_value(parameter, value):
    if parameter in {"top_p", "min_p", "repeat_penalty"}:
        return f"{float(value):.2f}" if float(value) == round(float(value), 2) else str(value)
    return str(value)


def menu(title, labels, default=None):
    print("\n" + title)
    for number, label in enumerate(labels, 1):
        print(f"{number}. {label}")
    while True:
        text = input(f"Choice (Enter = {default}): " if default else "Choice: ").strip()
        if not text and default is not None:
            return default
        if text.isdigit() and len(text) < 4 and 1 <= int(text) <= len(labels):
            return int(text)
        print("Choose one of the displayed numbers.")


def choose_parameter(title, excluded=None, advanced=False):
    names = list(PRESETS) if not advanced else [k for k in PARAMETERS if k != "seed"]
    while True:
        choice = menu(title, names + ["Back"])
        if choice == len(names) + 1:
            return None
        parameter = names[choice - 1]
        if parameter != excluded:
            return parameter
        print("Choose a different parameter for the second axis.")


def choose_range(parameter):
    # Extra advanced parameters have no diagnostic presets; enter their ranges explicitly.
    if parameter not in PRESETS:
        return ask_values(parameter)
    ranges = [PRESETS[parameter], *RANGE_PRESETS[parameter]]
    labels = ["Standard diagnostic range", "Lower range", "Middle range", "Upper range", "Fine range around baseline"]
    for label, values in zip(labels, ranges):
        print(f"{label}: {values}")
    choice = menu("Choose range:", labels + ["Enter custom values", "Back"])
    if choice == 7:
        return None
    return ask_values(parameter) if choice == 6 else ranges[choice - 1].copy()


def choose_seeds(single=False, current_seed=None):
    if single:
        choice = menu("Choose fixed seed:", ["Seed 1", "Seed 2", "Seed 3", "Seed 4", "Seed 5", "Enter custom seed", "Back"])
        if choice == 7:
            return None
        if choice <= 5:
            return [choice]
        while True:
            values = ask_values("seed")
            if values is None or len(values) == 1:
                return values
            print("Enter exactly one seed.")
    choice = menu("Choose seeds:", ["Seeds 1-5", "Seeds 1-3", "Seeds 1-10", "Current seed only" if current_seed is not None else "Single seed 1", "Enter custom seeds", "Back"])
    if choice == 6:
        return None
    return ask_values("seed") if choice == 5 else [list(range(1, 6)), [1, 2, 3], list(range(1, 11)), [current_seed if current_seed is not None else 1]][choice - 1]


def choose_experiment(mode, prompt, custom=False, model=MODEL, current_seed=1, baseline=None):
    parameter = choose_parameter("Choose X-axis parameter:" if mode == 2 else "Choose parameter:", advanced=custom)
    if parameter is None:
        return None
    y_parameter = None
    y_values = None
    if mode == 2:
        y_parameter = choose_parameter("Choose Y-axis parameter:", excluded=parameter, advanced=custom)
        if y_parameter is None:
            return None
    defaults = INTERACTION_RANGES if mode == 2 else PRESETS
    values = ask_values(parameter) if custom else defaults[parameter].copy()
    if values is None:
        return None
    if y_parameter:
        y_values = ask_values(y_parameter) if custom else defaults[y_parameter].copy()
        if y_values is None:
            return None
    seeds = MATRIX_SEEDS.copy() if mode == 1 else None
    fixed_seed = current_seed
    while True:
        title = {1: "Multi-seed sweep", 2: "Two-parameter grid", 3: "Quick sweep"}[mode]
        print(f"\n{title}\nModel: {model}\nPrompt: {prompt}\nX: {parameter} = {values}")
        if y_parameter:
            print(f"Y: {y_parameter} = {y_values}")
        print(f"Seeds: {seeds}" if seeds is not None else f"Seed: {fixed_seed}")
        print(f"Preset baseline: {json.dumps(baseline or {})}")
        print("Only the displayed axes and seeds are overridden; other unset parameters use backend defaults.")
        print(f"Generations: {len(values) * (len(y_values) if y_parameter else len(seeds) if seeds is not None else 1)}")
        labels = (["Run with these defaults", "Change X range", "Change Y range", "Change seed", "Back"] if mode == 2
                  else ["Run with these defaults" if mode == 1 else "Run with defaults", "Change range", "Change seeds" if mode == 1 else "Change seed", "Back"])
        action = menu("", labels, default=1)
        if action == 1:
            return parameter, values, seeds, fixed_seed, y_parameter, y_values
        if action == len(labels):
            return None
        if action == 2:
            replacement = choose_range(parameter)
            if replacement is not None:
                values = replacement
        elif mode == 2 and action == 3:
            replacement = choose_range(y_parameter)
            if replacement is not None:
                y_values = replacement
        else:
            replacement = choose_seeds(single=mode != 1)
            if replacement is not None:
                if mode == 1:
                    seeds = replacement
                else:
                    fixed_seed = replacement[0]


def run_experiment(opener, url, key, prompt, parameter, values, seeds, fixed_seed=1, y_parameter=None, y_values=None, model=MODEL, baseline=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    rows = []
    for settings in experiment_settings(parameter, values, seeds, fixed_seed, y_parameter, y_values):
        value = settings[parameter]
        timestamp = datetime.now(timezone.utc).isoformat()
        response, error = generate(opener, url, key, settings, prompt, model=model)
        rows.append({
            "timestamp": timestamp, "model": model, "prompt": prompt,
            "sweep_parameter": parameter, "sweep_value": value,
            **{k: (baseline or {}).get(k, "") for k in PARAMETERS},
            **settings, "response": response, "error": error,
            "preset_baseline": json.dumps(baseline or {}),
            "experiment_overrides": json.dumps(settings),
        })
    try:
        csv_path, html_path = write_reports(rows, stamp, parameter, values, seeds, prompt, fixed_seed, y_parameter, y_values, model=model, baseline=baseline)
    except OSError:
        print("Could not write reports; check results directory permissions and disk space.", file=sys.stderr)
        return 1
    failures = sum(bool(row["error"]) for row in rows)
    print(f"Completed {len(rows)} generations: {len(rows) - failures} succeeded, {failures} failed.")
    print(f"CSV: {csv_path}\nHTML: {html_path}")
    return 1 if failures else 0



def change_model(opener, url, key, current_model):
    """Read only model IDs and display names; never inspect preset configuration."""
    request = urllib.request.Request(
        url + "/api/models", headers={"Authorization": "Bearer " + key},
    )
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as reply:
            data = json.load(reply)
        entries = data.get("data") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError()
        models = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            identifier = entry.get("id")
            if not isinstance(identifier, str) or not identifier or not identifier.isprintable():
                continue
            name = entry.get("name")
            # Display only public model identifiers/names, never nested metadata.
            label = identifier
            if isinstance(name, str) and name.isprintable() and name != identifier:
                label += f" — {name}"
            models[identifier] = label
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            print(f"HTTP {exc.code}: model discovery authentication or access denied.")
        else:
            print(f"HTTP {exc.code}: could not retrieve models.")
        return current_model
    except (TimeoutError, socket.timeout):
        print("Model discovery timed out; current model unchanged.")
        return current_model
    except urllib.error.URLError:
        print("Unable to reach Open WebUI; current model unchanged.")
        return current_model
    except Exception:
        print("Could not read the model list; current model unchanged.")
        return current_model
    identifiers = list(models)
    choice = menu(f"Choose model:\nCurrent model: {current_model}",
                  list(models.values()) + ["Enter model ID manually", "Back"])
    if choice == len(identifiers) + 2:
        return current_model
    if choice <= len(identifiers):
        return identifiers[choice - 1]
    identifier = input("Exact model ID (Enter to keep current): ").strip()
    if identifier in models:
        return identifier
    if identifier:
        print("That ID was not found in /api/models; current model unchanged.")
    return current_model


# Safe generation metadata for documentation only; never copied into requests.
# Free-form content is redacted and private/system fields are excluded.
SAMPLING_NAMES = {
    "repeat_last_n", "tfs_z", "typical_p", "mirostat", "mirostat_tau",
    "mirostat_eta", "max_tokens", "stop",
}


def preset_generation_settings(params):
    extras = {}
    def inspect(values, prefix=""):
        for name, value in values.items():
            if name in {"options", "custom_params"} and isinstance(value, dict):
                inspect(value, prefix + name + ".")
                continue
            if name not in PARAMETERS and name not in SAMPLING_NAMES and name not in {"think", "format", "reasoning_effort", "num_keep", "num_batch", "num_gpu", "num_thread"}:
                continue
            numeric = isinstance(value, (int, float, bool)) and math.isfinite(value)
            if isinstance(value, str):
                try:
                    number = float(value)
                    if math.isfinite(number):
                        value, numeric = number, True
                except ValueError:
                    pass
            # Stop sequences and other free-form content may contain private text.
            shown = value if numeric or value is None else "[configured; non-numeric content withheld]"
            extras[prefix + name] = shown
    inspect(params)
    return extras


def read_preset_baseline(opener, url, key, model):
    request = urllib.request.Request(
        url + "/api/v1/models/model?" + urllib.parse.urlencode({"id": model}),
        headers={"Authorization": "Bearer " + key},
    )
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as reply:
            preset = json.load(reply)
        # Open WebUI can return null for a base model without a saved preset.
        if preset is None:
            return {}
        if not isinstance(preset, dict) or "params" not in preset:
            raise ValueError()
        params = preset["params"]
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError()
        baseline = preset_generation_settings(params)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: unable to inspect saved sampling settings; experiment cancelled.")
        return None
    except Exception:
        print("Unable to inspect saved sampling settings; experiment cancelled.")
        return None
    return baseline


def change_seed():
    while True:
        text = input("Enter seed number: ").strip()
        try:
            values = parse_values(text, "seed")
            if len(values) == 1:
                return values[0]
        except (ValueError, OverflowError):
            pass
        print("Enter one nonnegative integer.")


def load_suite():
    path = ROOT / "prompt_suite.txt"
    try:
        if not path.exists():
            path.write_text("# Add one complete prompt per line. Blank lines and comments are ignored.\n", encoding="utf-8")
            print(f"Created {path}. Add prompts using Edit prompt list.")
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    except (OSError, UnicodeError):
        print("Could not read prompt_suite.txt; check its permissions and UTF-8 encoding.")
        return []


def edit_suite():
    kate = shutil.which("kate")
    editor = kate or shutil.which("xdg-open")
    if editor is None:
        print(f"No editor launcher found. Edit {ROOT / 'prompt_suite.txt'} manually.")
        return
    try:
        # Kate --block waits for the document to close, including an existing instance.
        command = [editor, "--block"] if kate else [editor]
        subprocess.run(command + [str(ROOT / "prompt_suite.txt")], check=True)
        input("Save your changes, then press Enter to reload the prompt list: ")
    except (OSError, subprocess.CalledProcessError):
        print("Could not open the editor; edit prompt_suite.txt manually.")


def write_suite_reports(rows, stamp, model, baseline, prompts, seeds):
    RESULTS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.strptime(stamp, "%Y%m%dT%H%M%S_%fZ").strftime("%Y%m%d_%H%M%S_%f")
    basename = "prompt_suite_multiseed_" + timestamp
    csv_path = RESULTS / (basename + ".csv")
    html_path = RESULTS / (basename + ".html")
    fields = ["timestamp", "model", "prompt_index", "prompt", *PARAMETERS,
              "preset_baseline", "experiment_overrides", "response", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    safe = lambda value: escape(str(value), quote=True)
    cells = {}
    for row in rows:
        cells.setdefault((row["prompt_index"], row["seed"]), []).append(row)
    parts = ['<div class="matrix-scroll" role="region" aria-label="Behavior Testing responses" tabindex="0">',
             f'<table class="matrix" style="min-width:{260 + 260 * len(seeds)}px">'
             '<colgroup><col style="width:260px">' + '<col>' * len(seeds)
             + '</colgroup><thead><tr><th scope="col">Prompt</th>']
    parts.extend(f'<th scope="col">Seed {safe(seed)}</th>' for seed in seeds)
    parts.append('</tr></thead><tbody>')
    for index, prompt in enumerate(prompts, 1):
        parts.append(f'<tr><th scope="row">{safe(prompt)}</th>')
        for seed in seeds:
            parts.append('<td>')
            for row in cells.get((index, seed), []):
                parts.append(f'<small>{safe(row["timestamp"])}</small>')
                if row["error"]:
                    parts.append(f'<p class="error">{safe(row["error"])}</p>')
                parts.append(f'<pre>{safe(row["response"])}</pre>')
            parts.append('</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    html_path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Behavior Testing</title><style>' + report_css(True) + '</style>'
        '<h1>Behavior Testing</h1>'
        f'<p>Model: <strong>{safe(model)}</strong> · UTC run: {safe(stamp)}</p>'
        f'<p>Preset baseline: {safe(json.dumps(baseline))}</p>'
        f'<p>Seeds: {safe(seeds)} · Prompts: {len(prompts)} · Generations: {len(rows)}</p>'
        '<p>Only seed is overridden. Every prompt/seed combination uses an independent fresh '
        'user-only context. Other settings are inherited from the selected preset; unspecified '
        'settings use backend defaults. Private free-form preset content is withheld.</p>'
        + ''.join(parts) + '</html>', encoding="utf-8")
    return csv_path, html_path


def run_suite(opener, url, key, model, baseline, prompts, seeds):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    rows = []
    for index, prompt in enumerate(prompts, 1):
        for seed in seeds:
            timestamp = datetime.now(timezone.utc).isoformat()
            response, error = generate(opener, url, key, {"seed": seed}, prompt, model=model)
            rows.append({"timestamp": timestamp, "model": model, "prompt_index": index,
                         "prompt": prompt, **{k: baseline.get(k, "") for k in PARAMETERS},
                         "seed": seed, "preset_baseline": json.dumps(baseline),
                         "experiment_overrides": json.dumps({"seed": seed}),
                         "response": response, "error": error})
    try:
        csv_path, html_path = write_suite_reports(rows, stamp, model, baseline, prompts, seeds)
    except OSError:
        print("Could not write reports; check results directory permissions and disk space.")
        return
    failures = sum(bool(row["error"]) for row in rows)
    print(f"Completed {len(rows)} generations: {len(rows) - failures} succeeded, {failures} failed.")
    print(f"CSV: {csv_path}\nHTML: {html_path}")


def select_session_model(opener, url, key, model):
    selected = change_model(opener, url, key, model)
    if selected != model and read_preset_baseline(opener, url, key, selected) is not None:
        return selected
    return model


def prompt_suite(opener, url, key, model, current_seed, seeds):
    while True:
        prompts = load_suite()
        title = (f"BEHAVIOR TESTING\n\nCurrent model: {model}\nPrompts loaded: {len(prompts)}\n"
                 f"Seeds: {', '.join(map(str, seeds))}\nGenerations: {len(prompts) * len(seeds)}")
        choice = menu(title, ["Run tests", "Preview prompts", "Edit prompts", "Change seeds", "Change model", "Back"])
        if choice == 6:
            return model, seeds
        if choice == 5:
            model = select_session_model(opener, url, key, model)
        elif choice == 3:
            edit_suite()
        elif choice == 4:
            replacement = choose_seeds(current_seed=current_seed)
            if replacement is not None:
                seeds = list(dict.fromkeys(replacement))
        elif choice == 2:
            for index, prompt in enumerate(prompts, 1):
                print(f"{index}. {prompt}")
        elif not prompts:
            print("No usable prompts. Add prompts to prompt_suite.txt before running.")
        else:
            baseline = read_preset_baseline(opener, url, key, model)
            if baseline is None:
                continue
            print(f"Model: {model}\nPrompts: {len(prompts)}\nSeeds: {seeds}\nGenerations: {len(prompts) * len(seeds)}")
            if menu("Confirm Behavior Testing", ["Run", "Cancel"]) == 1:
                run_suite(opener, url, key, model, baseline, prompts, seeds)


def parameter_testing(opener, url, key, model, prompt, current_seed):
    while True:
        choice = menu(f"PARAMETER TESTING\n\nCurrent model: {model}\nCurrent prompt: {prompt}\nCurrent seed: {current_seed}\n",
                      ["Multi-seed sweep", "Two-parameter grid", "Quick sweep", "Custom experiment", "Change model", "Change prompt", "Change seed", "Back"])
        if choice == 8:
            return model, prompt, current_seed
        if choice == 7:
            current_seed = change_seed()
            continue
        if choice == 5:
            model = select_session_model(opener, url, key, model)
            continue
        if choice == 6:
            print(f"Current prompt:\n{prompt}\n")
            prompt = input("Enter new prompt: ").strip() or prompt
            continue
        custom = choice == 4
        if custom:
            choice = menu("Custom experiment", ["Custom multi-seed sweep", "Custom two-parameter grid", "Custom quick sweep", "Back"])
            if choice == 4:
                continue
        baseline = read_preset_baseline(opener, url, key, model)
        if baseline is None:
            continue
        experiment = choose_experiment(choice, prompt, custom, model=model, current_seed=current_seed, baseline=baseline)
        if experiment is not None:
            run_experiment(opener, url, key, prompt, *experiment, model=model, baseline=baseline)


def main():
    try:
        url, key = read_config()
    except (OSError, ValueError):
        print("Configuration error: check .env and the experiment settings.", file=sys.stderr)
        return 1
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    model, prompt, current_seed = MODEL, PROMPT, 1
    behavior_seeds = MATRIX_SEEDS.copy()
    try:
        while True:
            choice = menu("OpenWebUI Sampler Lab\n\nChoose mode:",
                          ["Parameter Testing\n   Compare how sampling settings change model behavior.\n",
                           "Behavior Testing\n   Test the current model across multiple independent prompts.\n",
                           "Quit"])
            if choice == 3:
                return 0
            if choice == 1:
                model, prompt, current_seed = parameter_testing(opener, url, key, model, prompt, current_seed)
            else:
                model, behavior_seeds = prompt_suite(opener, url, key, model, current_seed, behavior_seeds)
    except (EOFError, KeyboardInterrupt):
        print("\nSession ended.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
