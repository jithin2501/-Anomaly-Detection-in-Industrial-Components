"""Train PatchCore (builds the memory bank) and save model.ckpt.
Run from project root:  python -m ai_engine.train
"""
import shutil

from anomalib.engine import Engine
from anomalib.models import Patchcore

try:
    from anomalib.data import MVTecAD          # anomalib >= 2.0
except ImportError:
    from anomalib.data import MVTec as MVTecAD  # anomalib 1.x

from .config import (BACKBONE, CATEGORY, CORESET_RATIO, DATA_ROOT, LAYERS,
                     MODEL_PATH, WEIGHTS_DIR)


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    datamodule = MVTecAD(
        root=str(DATA_ROOT),
        category=CATEGORY,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )
    model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        pre_trained=True,
        coreset_sampling_ratio=CORESET_RATIO,
    )
    engine = Engine(default_root_dir=str(WEIGHTS_DIR))
    engine.fit(model=model, datamodule=datamodule)   # builds the memory bank
    engine.test(model=model, datamodule=datamodule)  # prints AUROC metrics

    ckpts = sorted(WEIGHTS_DIR.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    ckpts = [c for c in ckpts if c != MODEL_PATH]
    if not ckpts:
        raise SystemExit("No checkpoint found after training.")
    shutil.copy(ckpts[-1], MODEL_PATH)
    print(f"Saved checkpoint -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
