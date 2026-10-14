import requests
import base64
import io
import os
from PIL import Image

# === CONFIGURAZIONE ===
CARTELLA_INPUT = "dataset"
CARTELLA_OUTPUT = "dataset_caravaggio"  # nuova cartella: non salta i vecchi B/N

# Nome esatto del checkpoint (senza .safetensors)
NOME_CHECKPOINT = "juggernautXL_ragnarokBy"

# Nome esatto della LoRA (senza .safetensors)
NOME_LORA = "caravaggio_XL_resize"
PESO_LORA = 1.0  # consigliato dall'autore; prova anche 0.7 e 1.3

PROMPT = (
    f"<lora:{NOME_LORA}:{PESO_LORA}>, baroque oil painting, "
    f"chiaroscuro, tenebrism, visible brushstrokes, "
    f"warm earthy colors, deep reds, ochre, brown tones"
)

NEGATIVE_PROMPT = (
    "monochrome, grayscale, black and white, desaturated, "
    "photo, photorealistic, digital art, 3d render, anime, smooth skin, "
    "deformed hands, extra fingers, bad anatomy"
)

DENOISING_STRENGTH = 0.6
TARGET_PX = 1024   # scendi a 896 se vedi errori di memoria (OOM)
STEPS = 28
CFG = 6.0


# === UTILITIES ===

def verifica_server() -> bool:
    try:
        r = requests.get("http://127.0.0.1:7860/sdapi/v1/sd-models", timeout=5)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def carica_checkpoint(nome: str) -> bool:
    try:
        payload = {"sd_model_checkpoint": nome}
        r = requests.post("http://127.0.0.1:7860/sdapi/v1/options", json=payload, timeout=180)
        return r.status_code == 200
    except Exception as e:
        print(f"❌ Errore cambio checkpoint: {e}")
        return False


def immagine_in_base64(path: str) -> str:
    """Carica l'immagine a COLORI (RGB) e la codifica in base64."""
    img = Image.open(path).convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def calcola_dimensioni(path: str, target: int = 1024):
    img = Image.open(path)
    w, h = img.size
    ratio = w / h

    if ratio >= 1:
        out_w = target
        out_h = int(target / ratio)
    else:
        out_h = target
        out_w = int(target * ratio)

    out_w = max(512, (out_w // 8) * 8)
    out_h = max(512, (out_h // 8) * 8)
    return out_w, out_h


# === TRASFORMAZIONE ===

def trasforma_immagine(path_input: str, path_output: str) -> bool:
    url = "http://127.0.0.1:7860/sdapi/v1/img2img"

    out_w, out_h = calcola_dimensioni(path_input, target=TARGET_PX)
    img_b64 = immagine_in_base64(path_input)

    payload = {
        "init_images": [img_b64],
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "denoising_strength": DENOISING_STRENGTH,
        "steps": STEPS,
        "cfg_scale": CFG,
        "width": out_w,
        "height": out_h,
        "sampler_name": "DPM++ 2M Karras",
        "restore_faces": False,
        "tiling": False,
        "seed": -1,
    }

    print(f"   ↳ {out_w}x{out_h} | colore | denoising: {DENOISING_STRENGTH}")

    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        r = response.json()

        if "images" not in r or not r["images"]:
            print(f"   ❌ Risposta inattesa: {r}")
            return False

        image_data = base64.b64decode(r["images"][0])
        img = Image.open(io.BytesIO(image_data))
        img.save(path_output, format="PNG")
        print(f"   ✅ Salvata: {path_output}")
        return True

    except requests.exceptions.Timeout:
        print("   ❌ Timeout, prova a ridurre TARGET_PX o gli steps")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Errore di rete: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return False


# === MAIN ===

def processa_cartella():
    print("🔌 Verifica connessione...")
    if not verifica_server():
        print("❌ Server non raggiungibile su http://127.0.0.1:7860")
        print("   Avvia Forge con il flag --api")
        return

    print("✅ Server connesso!\n")

    print(f"📦 Caricando checkpoint: {NOME_CHECKPOINT}...")
    if not carica_checkpoint(NOME_CHECKPOINT):
        print("⚠️  Impossibile cambiare checkpoint via API, selezionalo manualmente in Forge")
    else:
        print("✅ Checkpoint caricato!\n")

    if not os.path.isdir(CARTELLA_INPUT):
        print(f"❌ Cartella '{CARTELLA_INPUT}' non trovata!")
        return

    os.makedirs(CARTELLA_OUTPUT, exist_ok=True)

    estensioni = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    immagini = [f for f in os.listdir(CARTELLA_INPUT) if f.lower().endswith(estensioni)]

    if not immagini:
        print(f"❌ Nessuna immagine trovata in '{CARTELLA_INPUT}'!")
        return

    print(f"📁 Trovate {len(immagini)} immagini da elaborare\n")

    successi, errori = 0, 0

    for i, nome_file in enumerate(immagini):
        path_input = os.path.join(CARTELLA_INPUT, nome_file)
        nome_output = f"{os.path.splitext(nome_file)[0]}_caravaggio.png"
        path_output = os.path.join(CARTELLA_OUTPUT, nome_output)

        if os.path.exists(path_output):
            print(f"[{i+1}/{len(immagini)}] ⏭️  Saltata (già esistente): {nome_file}")
            successi += 1
            continue

        print(f"[{i+1}/{len(immagini)}] 🎨 {nome_file}...")
        ok = trasforma_immagine(path_input, path_output)
        successi += ok
        errori += not ok

    print(f"\n{'='*50}")
    print(f"🎉 Completato! {successi} generate, {errori} errori")
    print(f"📂 Output in: '{CARTELLA_OUTPUT}'")


if __name__ == "__main__":
    processa_cartella()