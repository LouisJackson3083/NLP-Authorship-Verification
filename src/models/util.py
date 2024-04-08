from pytorch_lightning.callbacks import ModelCheckpoint


def build_checkpoint_callback(save_top_k,
                              filename="QTag-{epoch:02d}-{val_loss:.2f}",
                              monitor="val_loss",
                              mode="min"):
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        filename=filename,
        save_top_k=save_top_k,
        mode=mode,  # mode of the monitored quantity for optimization
    )
    return checkpoint_callback
