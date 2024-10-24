import torch
import pytorch_lightning as pl
from model import NN
from dataset import MnistDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v0")
    strategy = DeepSpeedStrategy()
    profiler = PyTorchProfiler(
        on_trace=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20)
    )
    model = NN(input_size=config.INPUT_SIZE, num_classes=config.NUM_CLASSES)
    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    trainer = pl.Trainer(
        strategy=strategy,
        profiler='simple',
        logger=logger,
        accelerator="gpu",
        devices=1,
        min_epochs=1,
        max_epochs=3,
        precision=16,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)

# Notes: Sanity check
# overfit_batches: before model training: check if we can overfit on single batch
# fast_dev_run: runs an entire pipeline of train val and test before training starts
# strategy: distributed training strategies
# profiler: tells you what is bottlenecking your code
