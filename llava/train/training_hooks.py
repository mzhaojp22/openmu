import os
from transformers import TrainerCallback, is_tensorboard_available
import logging

# API ref: https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/callback#transformers.TrainerCallback


def custom_rewrite_logs(d, mode):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if mode == "eval" and k.startswith(eval_prefix):
            if k[eval_prefix_len:] == "loss":
                new_d["combined/" + k[eval_prefix_len:]] = v
        elif mode == "test" and k.startswith(test_prefix):
            if k[test_prefix_len:] == "loss":
                new_d["combined/" + k[test_prefix_len:]] = v
        elif mode == "train":
            if k == "loss":
                new_d["combined/" + k] = v
    return new_d


class DetailedLoggingTensorBoardCallback(TrainerCallback):

    def __init__(self, tb_writers=None):
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard:
            raise RuntimeError("Need to install tensorboardX.")
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writers = tb_writers

    def _init_summary_writer(self, args, log_dir=None):
        # args.logging_dir default to *output_dir/runs/CURRENT_DATETIME_HOSTNAME
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writers = dict(
                train=self._SummaryWriter(log_dir=os.path.join(log_dir, "train")),
                eval=self._SummaryWriter(log_dir=os.path.join(log_dir, "eval")),
            )

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writers is None:
            self._init_summary_writer(args, log_dir)

        for k, tbw in self.tb_writers.items():
            # (train, tbw), (eval, tbw)
            tbw.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    tbw.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(tbw, "add_hparams"):
                tbw.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        tb_writer = self.tb_writers["train"]
        tb_writer.add_scalar(
            "global_grad_norm", state._GLOBAL_STEP_NORM, state.global_step
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writers is None:
            self._init_summary_writer(args)

        for tbk, tbw in self.tb_writers.items():
            # (train, tbw), (eval, tbw)
            logs_new = custom_rewrite_logs(logs, mode=tbk)
            for k, v in logs_new.items():
                if isinstance(v, (int, float)):
                    tbw.add_scalar(k, v, state.global_step)
                else:
                    logging.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            tbw.flush()

    def on_train_end(self, args, state, control, **kwargs):
        for tbw in self.tb_writers.values():
            tbw.close()
        self.tb_writers = None
