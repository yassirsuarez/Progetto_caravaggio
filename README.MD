# Progetto Caravaggio
 
Pipeline per costruire un dataset di **volti da dipinti di Caravaggio** (e da immagini generate con Stable Diffusion nel suo stile) e per analizzarlo con tecniche di **clustering su embedding**, con l'obiettivo di individuare volti ricorrenti (*Fillide*, dal nome di una delle modelle di Caravaggio).
 
## Obiettivo
 
 Dato un volto di test (`Unknown.jpg`), verificare se compare in cluster di volti simili tra i dipinti di Caravaggio, confrontando due approcci: embedding **facciali** (ArcFace) ed embedding **visivi generici** (DINOv2). Le immagini generate con IA servono a mettere alla prova i modelli, verificandone la robustezza.
 
## Struttura del progetto
 
```
progetto_caravaggio/
├── dataset/
│   ├── originali/
│   │   └── caravaggio/                # Dipinti originali
│   ├── generative_ia/
│   │   ├── dataset/                   # Immagini di input per la generazione
│   │   ├── dataset_caravaggio/        # Output generati con Stable Diffusion 1.5
│   │   ├── dataset_caravaggio_nuovo/  # Output generati con SDXL
│   │   └── codice.py                  # Script di generazione con Forge
│   ├── pre_processato/
│   │   ├── colorati/{normali, generative_ia}/
│   │   └── bianco_e_nero/{normali, generative_ia}/
│   ├── genere/
│   │   ├── colorati/{Male, Female}/
│   │   └── bianco_e_nero/{Male, Female}/
│   └── pre_processing.ipynb           # Ritaglio volti, B/N, smistamento per genere
└── script/
    ├── Fillide_arcface.ipynb          # Clustering con embedding ArcFace
    ├── Fillide_dino.ipynb             # Clustering con embedding DINOv2
    └── requirements_arcface.txt       # Dipendenze per ArcFace
```
 
## Flusso di lavoro
 
### 1. Generazione delle immagini IA
 
Script: `dataset/generative_ia/codice.py`
 
Trasforma le immagini di input in dipinti in stile Caravaggio tramite l'API di **Stable Diffusion Forge** (img2img). Sono state usate due configurazioni di modelli, una per ciascuna generazione del dataset.
 
Requisiti comuni:
 
- Forge avviato con il flag `--api` (server su `http://127.0.0.1:7860`)
- Checkpoint in `models/Stable-diffusion` e LoRA in `models/Lora`
- GPU con almeno 8 GB di VRAM
```bash
python codice.py
```
 
Lo script salta le immagini già generate, quindi può essere interrotto e ripreso.
 
#### Prima versione: Stable Diffusion 1.5
 
Output in `dataset_caravaggio/`.
 
| Parametro | Valore |
|---|---|
| Checkpoint | `v1-5-pruned-emaonly` (Stable Diffusion 1.5) |
| LoRA | `CARAVAGGIO V1-000002` (peso 0.85) |
| Input | Immagine convertita in bianco e nero |
| Risoluzione | 512 px sul lato lungo |
| Denoising | 0.65 |
| Steps / CFG | 35 / 7.5 |
| Sampler | DPM++ 2M Karras |
| Ripristino volti | Disattivato (`restore_faces: False`), per evitare volti "di plastica" |
 
Prompt: pittura barocca del Seicento, chiaroscuro direzionale marcato, tenebrismo, toni caldi della pelle, colori terrosi, pennellate visibili e impasto. Il prompt negativo esclude look fotorealistico, 3D, anime, pelle liscia e colori neon.
 
#### Seconda versione: SDXL
 
Output in `dataset_caravaggio_nuovo/`.
 
| Parametro | Valore |
|---|---|
| Checkpoint | `juggernautXL_ragnarokBy` (SDXL) |
| LoRA | `caravaggio_XL_resize` (peso 1.0) |
| Risoluzione | 1024 px sul lato lungo |
| Denoising | 0.6 |
| Steps / CFG | 28 / 6.0 |
| Sampler | DPM++ 2M Karras |
 
Per i problemi di memoria con 8 GB di VRAM, porta `TARGET_PX` a 896.
 
### 2. Pre-processing dei volti
 
Notebook: `dataset/pre_processing.ipynb`
 
1. **Ritaglio e allineamento**: DeepFace con detector RetinaFace estrae e allinea i volti.
2. **Conversione in bianco e nero**: crea la versione in scala di grigi di ogni volto.
3. **Smistamento per genere**: DeepFace stima il genere e copia i file in `Male` / `Female`.
Per scegliere il dataset da elaborare (originali o generative_ia) commenta e decommenta le coppie `input_folder` / `output_folder` nelle celle.
 
Note:
 
- Con `enforce_detection=False`, se non viene trovato nessun volto viene salvata l'intera immagine: controlla a campione i ritagli.
- La stima del genere è automatica e può sbagliare sui dipinti.
### 3. Analisi con clustering
 
Entrambi i notebook lavorano sui volti smistati per genere in `dataset/genere/`, in bianco e nero o a colori, e seguono la stessa logica. Il dataset da analizzare si sceglie cambiando il percorso nella cella di configurazione (`dataset_bn` in ArcFace, `dataset_path` in DINOv2):

| Variante | Percorso |
|---|---|
| Femminili in bianco e nero | `dataset/genere/bianco_e_nero/Female` |
| Femminili a colori | `dataset/genere/colorati/Female` |
| Maschili in bianco e nero | `dataset/genere/bianco_e_nero/Male` |
| Maschili a colori | `dataset/genere/colorati/Male` |
 
1. estrazione di un embedding per ogni immagine;
2. riduzione a 2D con **UMAP** (`n_neighbors=15`, `min_dist=0.1`, `random_state=42`);
3. clustering con **HDBSCAN** e con **DBSCAN** (con k-distance plot per scegliere `eps`);
4. ricerca dell'immagine di test (`Unknown.jpg`): grafico dei cluster, galleria dei membri del suo cluster e **similarità coseno** con ciascuno.
| | `Fillide_arcface.ipynb` | `Fillide_dino.ipynb` |
|---|---|---|
| Modello | ArcFace ResNet50 (InsightFace `buffalo_l`) | DINOv2 `dinov2_vitb14` (torch.hub) |
| Cosa misura | Identità del volto | Somiglianza visiva generale |
| Embedding | 512 dimensioni | 768 dimensioni |
| Rilevamento volto | SCRFD; se non trova nulla, usa l'intera immagine a 112×112 | Nessuno: resize 256 + center crop 224 |
| HDBSCAN | `min_cluster_size=2` | `min_cluster_size=5` |
| DBSCAN | `eps=0.55`, `min_samples=2` | `eps=0.50`, `min_samples=3` |
| Extra | Colora i punti per **periodo stilistico** (dedotto dal nome del file) | — |
 
Il notebook ArcFace assegna il periodo cercando parole chiave nel nome del file:
 
1. Primo Periodo Romano
2. Periodo Grandi Patroni
3. Maturità Romana
4. Periodo Esilio Tardo
5. Sconosciuto / Altro
Per usarli:
 
1. Metti `Unknown.jpg` (o cambia `nome_test` / `TEST_FILE`) nella cartella analizzata.
2. Controlla il percorso `dataset_bn` / `dataset_path` nella cella di configurazione.
3. Guarda il k-distance plot e aggiusta `eps` di conseguenza.
Note metodologiche:
 
- Il clustering è fatto sulle coordinate **UMAP 2D**, non sugli embedding originali: i cluster dipendono anche dai parametri di UMAP, quindi non vanno letti come prova definitiva di identità.
- ArcFace è addestrato su volti fotografici: su dipinti in B/N le similarità sono meno affidabili. Il notebook segnala quante immagini hanno usato il fallback senza volto rilevato.
## Installazione
 
Si consiglia Python 3.10 in un ambiente virtuale.
 
**Generazione e pre-processing:**
 
```bash
pip install requests pillow opencv-python numpy imutils deepface retina-face
```
 
**Notebook ArcFace:**
 
```bash
pip install insightface onnxruntime-gpu opencv-python umap-learn hdbscan scikit-learn pandas matplotlib tqdm pillow
```
 
Senza GPU NVIDIA sostituisci `onnxruntime-gpu` con `onnxruntime`.
 
**Notebook DINOv2:** servono `torch`, `torchvision`, `umap-learn`, `hdbscan`, `scikit-learn`, `pandas`, `matplotlib`, `tqdm`, `pillow`. Al primo avvio `torch.hub` scarica il modello da GitHub, quindi serve una connessione internet.
 
Nei notebook di pre-processing viene impostata `TF_USE_LEGACY_KERAS=1` per la compatibilità di RetinaFace con Keras recente.
 
## Risultati

 
## Note sui dati
 
- Il dataset originale proviene da dipinti di Caravaggio in pubblico dominio.
- Le LoRA usate (`CARAVAGGIO V1-000002` per SD 1.5 e `caravaggio_XL_resize` per SDXL, da Civitai): verifica la licenza di LoRA e checkpoint prima di ridistribuire i risultati. Per `caravaggio_XL_resize` l'autore dichiara che il training usa immagini in pubblico dominio.
