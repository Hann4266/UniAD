import os
import os.path as osp
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger import LoggerHook

try:
    import wandb
except Exception:
    wandb = None


def is_main():
    return int(os.environ.get("RANK", "0")) == 0


@HOOKS.register_module()
class UniADWandbHook(LoggerHook):
    def __init__(self, project="intent_predict", name=None, interval=50, log_ckpt=True, **kwargs):
        super().__init__(interval=interval, **kwargs)   # <-- IMPORTANT
        self.project = project
        self.name = name
        self.log_ckpt = log_ckpt

    def before_run(self, runner):
        super().before_run(runner)  # optional but good practice

        if wandb is None:
            runner.logger.warning("wandb not installed; UniADWandbHook disabled.")
            return

        if is_main():
            wandb.init(project=self.project, name=self.name, config={"work_dir": runner.work_dir})
            wandb.log({"debug/started": 1}, step=0)
        else:
            wandb.init(mode="disabled")

    # <-- THIS IS THE KEY CHANGE
    def log(self, runner):
        if wandb is None or not is_main() or wandb.run is None:
            return

        out = runner.log_buffer.output
        step = runner.iter

        log_dict = {}
        for k, v in out.items():
            try:
                log_dict[f"train/{k}"] = float(v)
            except Exception:
                pass

        if not log_dict:
            log_dict = {"debug/empty_log_buffer": 1}

        wandb.log(log_dict, step=step)

    def after_train_epoch(self, runner):
        if not self.log_ckpt:
            return
        if wandb is None or wandb.run is None or not is_main():
            return
        latest = osp.join(runner.work_dir, "latest.pth")
        if osp.exists(latest):
            art = wandb.Artifact("model", type="checkpoint")
            art.add_file(latest)
            wandb.log_artifact(art)

    def after_run(self, runner):
        if wandb is not None and wandb.run is not None and is_main():
            wandb.finish()
        super().after_run(runner)  # optional