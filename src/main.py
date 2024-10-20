import os
import warnings
from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.plugins.layer_sync import TorchSyncBatchNorm

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg_dict["test"]["output_path"] = os.path.join("outputs", cfg_dict["output_dir"], "test")
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    else:  # for resuming
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["cwd"]) / "outputs" / cfg_dict.output_dir
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({"id": cfg_dict.wandb.id, "resume": "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    elif cfg_dict.use_tensorboard is not None:
        tensorboard_dir = output_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True, parents=True)
        logger = TensorBoardLogger(save_dir=output_dir)
        callbacks.append(LearningRateMonitor("step", True))
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",  # save the lastest k ckpt, can do offline test later
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = "_"

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=cfg.device,
        strategy="ddp",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        precision=cfg.precision,
    )
    print(f"GPU number is {torch.cuda.device_count()}")
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder, decoder)
    if cfg.mode == "train" and checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path)["state_dict"]
        ckpt = {".".join(k.split(".")[1:]): v for k, v in ckpt.items()}
        encoder.load_state_dict(ckpt)
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
    )
    model_wrapper = TorchSyncBatchNorm().apply(model_wrapper)

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        print("begin to train fit!")
        try:
            print(f"resume from {checkpoint_path}!!!")
            trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
        except:
            print(f"start from scratch!!!")
            trainer.fit(model_wrapper, datamodule=data_module)
    elif cfg.mode == "test":
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )
    else:
        raise NotImplementedError(f"The {cfg.mode} mode is not implemented!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")

    train()
