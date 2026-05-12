# =============================================================================
# parameters_dl.py
# Parameters for the Pijáčková deep learning comparison notebook.
# Kept separate from parameters.py so classical-ML pipelines are unaffected.
# =============================================================================

# ── Data ─────────────────────────────────────────────────────────────────────
deep_training_dir = '/home/uav-cyberlab-rfml/rww_conference/annotations/Training'

# ── IQ windowing ─────────────────────────────────────────────────────────────
WINDOW_LEN           = 256     # IQ samples per window  (was 128 — longer context helps N11/Ruko separation)
WINDOW_STRIDE        = 128     # 50% overlap            (was 64)
MAX_WINDOWS_PER_FILE = 2000    # cap per file

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE  = 128
EPOCHS      = 50
LR          = 7e-4
N_FOLDS     = 5
PATIENCE    = 15               # was 8 — give models more room before early stop

# ── MCTransformer ─────────────────────────────────────────────────────────────
MC_EMBED_DIM  = 64
MC_NUM_HEADS  = 4
MC_FF_DIM     = 128            # was 16 — should be ~2x embed_dim

# ── Misc ─────────────────────────────────────────────────────────────────────
SAVE_METRICS_TO_FILE = True
TRAINING_DATASET     = 'rww_4class_drone_id'
