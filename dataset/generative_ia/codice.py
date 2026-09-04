import requests
import base64
import io
import os
from PIL import Image

# === CONFIGURAZIONE ===
CARTELLA_INPUT = "dataset"
CARTELLA_OUTPUT = "dataset_caravaggio"

# Nome esatto del checkpoint (senza .safetensors)
NOME_CHECKPOINT = "v1-5-pruned-emaonly"

# Nome esatto della LoRA (senza .safetensors)
NOME_LORA = "CARAVAGGIO V1-000002"
PESO_LORA = 0.85  # Spingiamo leggermente di più visto che partiamo dal B/N

# PROMPT POSITIVO: Enfatizziamo la pittura barocca, i pigmenti storici e il chiaroscuro materico
PROMPT = (
    f"<lora:{NOME_LORA}:{PESO_LORA}>, 17th century baroque oil painting by Caravaggio, "
    f"heavy directional chiaroscuro, dramatic light and shadows, tenebrism, warm skin tones, "
    f"earthy colors, visible organic brushstrokes, rich impasto texture, historical oil canvas patina"
)

# PROMPT NEGATIVO: Blindiamo l'IA per impedirle di lisciare i pixel e usare colori sintetici
NEGATIVE_PROMPT = (
    "photorealistic, digital art, 3d render, anime, smooth texture, airbrushed, plastic skin, "
    f"glossy, perfect studio lighting, neon, vibrant digital colors, cgi, vector, sharp digital focus"
)

# Avendo rimosso il colore, diamo più libertà di reinvenzione cromatica al modello (0.60-0.65)
DENOISING_STRENGTH = 0.65 


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
        r = requests.post("http://127.0.0.1:7860/sdapi/v1/options", json=payload, timeout=60)
        return r.status_code == 200
    except Exception as e:
        print(f"❌ Errore cambio checkpoint: {e}")
        return False


def immagine_in_base64_bn(path: str) -> str:
    """Carica l'immagine, la converte in Bianco e Nero e la codifica in base64."""
    img = Image.open(path)
    # Convertiamo in scala di grigi ('L') per dare massima libertà cromatica a SD
    img_bn = img.convert("L")
    
    # Se vuoi anche aumentare il contrasto via codice per forzare le ombre, puoi scommentare le 2 righe sotto:
    # from PIL import ImageEnhance
    # img_bn = ImageEnhance.Contrast(img_bn).enhance(1.3)
    
    buffered = io.BytesIO()
    img_bn.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def calcola_dimensioni(path: str, target: int = 512):
    img = Image.open(path)
    w, h = img.size
    ratio = w / h

    if ratio >= 1:
        out_w = target
        out_h = int(target / ratio)
    else:
        out_h = target
        out_w = int(target * ratio)

    out_w = (out_w // 8) * 8
    out_h = (out_h // 8) * 8
    return out_w, out_h


# === TRASFORMAZIONE ===

def trasforma_immagine(path_input: str, path_output: str) -> bool:
    url = "http://127.0.0.1:7860/sdapi/v1/img2img"

    out_w, out_h = calcola_dimensioni(path_input, target=512)
    
    # Usiamo la nuova funzione che converte l'immagine in Bianco e Nero
    img_b64 = immagine_in_base64_bn(path_input)

    payload = {
        "init_images": [img_b64],
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "denoising_strength": DENOISING_STRENGTH,
        "steps": 35,  # Alzato leggermente a 35 per definire meglio la micro-texture della tela
        "cfg_scale": 7.5,
        "width": out_w,
        "height": out_h,
        "sampler_name": "DPM++ 2M Karras",
        "restore_faces": False,  # CRITICO: Impostato su False per evitare facce 'di plastica' moderne
        "tiling": False,
        "seed": -1,
    }

    print(f"   ↳ {out_w}x{out_h} | B/N forzato | denoising: {DENOISING_STRENGTH}")

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        r = response.json()

        if "images" not in r or not r["images"]:
            print(f"   ❌ Risposta inattesa: {r}")
            return False

        image_data = base64.b64decode(r["images"][0])
        img = Image.open(io.BytesIO(image_data))
        img.save(path_output, format="PNG")
        print(f"   ✅ Salvata con texture barocca: {path_output}")
        return True

    except requests.exceptions.Timeout:
        print("   ❌ Timeout — prova a ridurre gli steps")
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
        print("⚠️  Impossibile cambiare checkpoint via API — selezionalo manualmente in Forge")
    else:
        print("✅ Checkpoint caricato!\n")

    if not os.path.isdir(CARTELLA_INPUT):
        print(f"❌ Cartella '{CARTELLA_INPUT}' non trovata!")
        return

    os.makedirs(CARTELLA_OUTPUT, exist_ok=True)

    estensioni = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    immagini = [
        f for f in os.listdir(CARTELLA_INPUT)
        if f.lower().endswith(estensioni)
    ]

    if not immagini:
        print(f"❌ Nessuna immagine trovata in '{CARTELLA_INPUT}'!")
        return

    print(f"📁 Trovate {len(immagini)} immagini da elaborare in B/N\n")

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
    print(f"🎉 Completato! {successi} generate con successo, {errori} errori")
    print(f"📂 Dataset IA generato in: '{CARTELLA_OUTPUT}'")


if __name__ == "__main__":
    processa_cartella()