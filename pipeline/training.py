import math
import time
from pathlib import Path
from collections.abc import Iterator
import torch
from pipeline.inference import generate_sample
from tqdm import tqdm
from utils.calculate_params import count_params
from utils.plot import save_training_plots

try:
    import wandb
except ImportError:
    wandb = None

@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=None, show_progress=False, desc="Evaluating"):
    model = model.to(device)
    model.eval()
    total_loss, total_tokens = 0.0, 0
    iterator = dataloader
    total = None
    if show_progress:
        total = min(len(dataloader), max_batches) if max_batches is not None else len(dataloader)
        iterator = tqdm(
            dataloader,
            total=total,
            desc=desc,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

    for batch_idx, (inp, lbl) in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inp, lbl = inp.to(device), lbl.to(device)
        _, loss = model(inp, labels=lbl)
        total_loss += loss.item() * lbl.numel()
        total_tokens += lbl.numel()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


def cosine_schedule(optimizer, step, total_steps, lr_max, lr_min=1e-5, warmup_steps=0, decay_start_step=None):
    if decay_start_step is None:
        decay_start_step = warmup_steps
    
    if step < warmup_steps:
        lr = lr_max * (step + 1) / max(warmup_steps, 1)
    elif step < decay_start_step:
        lr = lr_max  # ← flat di sini
    else:
        progress = (step - decay_start_step) / max(total_steps - decay_start_step, 1)
        progress = min(max(progress, 0.0), 1.0)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def _cycle_dataloader(dl) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in dl:
            yield batch


def _maybe_init_wandb(config, model_config, label):
    if not config.enabled:
        return None
    if wandb is None:
        raise ImportError("wandb belum terpasang, tapi wandb.enabled=true di config")
    return wandb.init(
        project=config.project,
        name=config.name or label,
        entity=config.entity,
        mode=config.mode,
        config={
            "label": label,
            "model": model_config,
        },
    )


def _resolve_amp_dtype(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def _autocast_enabled(device, precision: str) -> bool:
    return device.type == "cuda" and precision in {"fp16", "bf16"}


def _save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scaler,
    step,
    best_val_loss,
    best_step,
    history,
    label,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "label": label,
            "step": step,
            "best_val_loss": best_val_loss,
            "best_step": best_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "history": history,
        },
        checkpoint_path,
    )


def _build_optimizer(model, train_config):
    if getattr(train_config, "optimizer_name", "adamw").lower() != "adamw":
        raise ValueError("TrainingConfig hanya mendukung optimizer AdamW")
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
    )


def train(
    model,
    train_dl,
    val_dl,
    device,
    label,
    train_config,
    generation_config=None,
    tokenizer=None,
    sample_prompt=None,
    wandb_config=None,
    model_config=None,
    checkpoint_config=None,
    resume_checkpoint=None,
):
    model = model.to(device)
    opt = _build_optimizer(model, train_config)
    amp_enabled = _autocast_enabled(device, train_config.precision)
    amp_dtype = _resolve_amp_dtype(train_config.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and train_config.precision == "fp16"))
    run = _maybe_init_wandb(wandb_config, model_config, label) if wandb_config is not None else None
    checkpoint_dir = None
    if checkpoint_config is not None and checkpoint_config.enabled:
        checkpoint_dir = Path(checkpoint_config.output_directory)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if run is not None and wandb_config.log_model:
        run.watch(model, log="all", log_freq=train_config.log_every_steps)

    history = {
        "train_loss": [],
        "train_steps": [],
        "grad_norm": [],
        "grad_norm_steps": [],
        "learning_rates": [],
        "learning_rate_steps": [],
        "val_loss": [],
        "val_ppl": [],
        "steps": [],
        "samples": [],
        "elapsed_times": [],
    }
    start_step = 1
    best_val_loss = float("inf")
    best_step = 0
    early_stop_counter = 0
    best_state_dict = None

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        opt.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scaler_state = resume_checkpoint.get("scaler_state_dict")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        history = resume_checkpoint.get("history", history)
        best_val_loss = resume_checkpoint.get("best_val_loss", best_val_loss)
        best_step = resume_checkpoint.get("best_step", best_step)
        resumed_step = int(resume_checkpoint.get("step", 0))
        start_step = resumed_step + 1
        if train_config.restore_best_model and best_val_loss < float("inf"):
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    print(f"\n  ── Training [{label}] ──")
    print(f"  Optimizer: {train_config.optimizer_name.upper()}")
    if resume_checkpoint is not None:
        print(f"  Resume from step: {start_step - 1:,}")

    train_iter = _cycle_dataloader(train_dl)
    running_loss = 0.0
    running_grad_norm = 0.0
    window_count = 0
    resumed_elapsed = 0.0
    if resume_checkpoint is not None and history.get("elapsed_times"):
        resumed_elapsed = float(history["elapsed_times"][-1])
    started_at = time.time() - resumed_elapsed
    print(f"  Total steps: {train_config.max_steps:,}")
    if start_step > train_config.max_steps:
        print("  Resume checkpoint sudah mencapai atau melewati total max_steps, skip training.")
        history["best_val_loss"] = best_val_loss
        history["best_step"] = best_step
        if run is not None:
            run.finish()
        return history
    with tqdm(
        total=train_config.max_steps,
        initial=start_step - 1,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        mininterval=0.1,
        smoothing=0.05,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    ) as pbar:
        for step in range(start_step, train_config.max_steps + 1):
            if step == start_step:
                pbar.write("Starting training step loop...")
            inp, lbl = next(train_iter)
            inp, lbl = inp.to(device), lbl.to(device)

            lr = cosine_schedule(
                opt,
                step - 1,
                train_config.max_steps,
                train_config.learning_rate,
                train_config.min_learning_rate,
                train_config.warmup_steps,
                getattr(train_config, "decay_start_step", None),
            )

            model.train()
            opt.zero_grad()
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                _, loss = model(inp, labels=lbl)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(opt)
            scaler.update()

            loss_value = loss.item()
            grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            running_loss += loss_value
            running_grad_norm += grad_norm_value
            window_count += 1

            if run is not None:
                run.log(
                    {
                        "train/loss_step": loss_value,
                        "train/grad_norm_step": grad_norm_value,
                        "train/lr": lr,
                        "train/step": step,
                    },
                    step=step,
                )

            avg_train_loss = running_loss / max(window_count, 1)
            avg_grad_norm = running_grad_norm / max(window_count, 1)
            elapsed = time.time() - started_at
            pbar.set_postfix(
                step=f"{step}/{train_config.max_steps}",
                loss=f"{avg_train_loss:.4f}",
                grad=f"{avg_grad_norm:.3f}",
                lr=f"{lr:.2e}",
                best=f"{best_val_loss:.4f}" if best_val_loss < float("inf") else "-",
                patience=f"{early_stop_counter}/{train_config.early_stopping_patience}",
            )
            pbar.update(1)

            if step % train_config.log_every_steps == 0:
                history["train_loss"].append(avg_train_loss)
                history["train_steps"].append(step)
                history["grad_norm"].append(avg_grad_norm)
                history["grad_norm_steps"].append(step)
                history["learning_rates"].append(lr)
                history["learning_rate_steps"].append(step)
                tqdm.write(
                    f"[Train @ step {step}] "
                    f"loss={avg_train_loss:.4f} grad_norm={avg_grad_norm:.4f} lr={lr:.6f}"
                )
                if run is not None:
                    run.log(
                        {
                            "train/loss_window": avg_train_loss,
                            "train/grad_norm_window": avg_grad_norm,
                            "time/elapsed": elapsed,
                        },
                        step=step,
                    )
                running_loss = 0.0
                running_grad_norm = 0.0
                window_count = 0

            if step % train_config.sample_every_steps == 0 and tokenizer is not None and sample_prompt:
                sample_text = generate_sample(
                    model,
                    tokenizer,
                    device,
                    sample_prompt,
                    MAX_SEQ=getattr(model, "max_seq", None),
                    max_new_tokens=generation_config.max_new_tokens if generation_config is not None else 40,
                    temperature=generation_config.temperature if generation_config is not None else 0.9,
                    top_k=generation_config.top_k if generation_config is not None else 40,
                )
                history["samples"].append({"step": step, "prompt": sample_prompt, "text": sample_text})
                tqdm.write(f"[Sample @ step {step}] {sample_text}")
                if run is not None:
                    run.log(
                        {
                            "sample/step": step,
                            "sample/prompt": sample_prompt,
                            "sample/text": sample_text,
                        },
                        step=step,
                    )

            if step % train_config.eval_every_steps == 0 or step == train_config.max_steps:
                avg_train_loss = running_loss / max(window_count, 1) if window_count > 0 else loss_value
                avg_grad_norm = running_grad_norm / max(window_count, 1) if window_count > 0 else grad_norm_value
                tqdm.write(
                    f"[Eval start @ step {step}] "
                    f"running validation on up to "
                    f"{train_config.eval_max_batches or len(val_dl)} batches"
                )
                val_loss, val_ppl = evaluate(
                    model,
                    val_dl,
                    device,
                    max_batches=train_config.eval_max_batches,
                    show_progress=True,
                    desc=f"Eval@{step}",
                )
                elapsed = time.time() - started_at

                history["steps"].append(step)
                history["val_loss"].append(val_loss)
                history["val_ppl"].append(val_ppl)
                history["elapsed_times"].append(elapsed)

                tqdm.write(
                    f"[Eval @ step {step}] train_loss={avg_train_loss:.4f} "
                    f"grad_norm={avg_grad_norm:.4f} val_loss={val_loss:.4f} "
                    f"val_ppl={val_ppl:.2f} lr={lr:.6f}"
                )

                improved = val_loss < (best_val_loss - train_config.early_stopping_min_delta)
                if improved:
                    best_val_loss = val_loss
                    best_step = step
                    early_stop_counter = 0
                    if train_config.restore_best_model:
                        best_state_dict = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in model.state_dict().items()
                        }
                    if checkpoint_dir is not None and checkpoint_config.save_best:
                        _save_checkpoint(
                            checkpoint_dir / "best.pt",
                            model,
                            opt,
                            scaler,
                            step,
                            best_val_loss,
                            best_step,
                            history,
                            label,
                        )
                        tqdm.write(f"[Checkpoint] best -> {checkpoint_dir / 'best.pt'}")
                else:
                    early_stop_counter += 1

                if checkpoint_dir is not None and checkpoint_config.save_last:
                    _save_checkpoint(
                        checkpoint_dir / "last.pt",
                        model,
                        opt,
                        scaler,
                        step,
                        best_val_loss,
                        best_step,
                        history,
                        label,
                    )
                    tqdm.write(f"[Checkpoint] last -> {checkpoint_dir / 'last.pt'}")

                if checkpoint_dir is not None:
                    save_training_plots(
                        history=history,
                        output_dir=checkpoint_dir / "plots",
                        run_name=label,
                    )

                if run is not None:
                    run.log(
                        {
                            "eval/loss": val_loss,
                            "eval/ppl": val_ppl,
                            "train/loss_window": avg_train_loss,
                            "train/grad_norm_window": avg_grad_norm,
                            "time/elapsed": elapsed,
                            "early_stop/best_val_loss": best_val_loss,
                            "early_stop/best_step": best_step,
                            "early_stop/counter": early_stop_counter,
                        },
                        step=step,
                    )

                if train_config.early_stopping_patience > 0 and early_stop_counter >= train_config.early_stopping_patience:
                    tqdm.write(
                        f"Early stopping at step {step} "
                        f"(best step {best_step}, best val loss {best_val_loss:.4f})"
                    )
                    break

    if train_config.restore_best_model and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    history["best_val_loss"] = best_val_loss
    history["best_step"] = best_step
    if checkpoint_dir is not None:
        save_training_plots(
            history=history,
            output_dir=checkpoint_dir / "plots",
            run_name=label,
        )
    if run is not None:
        run.finish()

    return history
