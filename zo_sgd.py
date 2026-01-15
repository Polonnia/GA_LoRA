import torch
import torch.nn.functional as F
import numpy as np

from utils import *
from loralib.utils import (
    mark_only_lora_as_trainable, apply_lora, get_lora_parameters,
    save_lora
)

# -------------------------
# Helpers: flatten / unflatten LoRA params
# -------------------------
def _flatten_params(params):
    """Return a 1D tensor view of all params concatenated."""
    return torch.cat([p.data.view(-1) for p in params], dim=0)

def _flatten_grads_like(params):
    """Same shape as flatten_params, but for grad buffer (not used here)."""
    return torch.cat([torch.zeros_like(p.data).view(-1) for p in params], dim=0)

def _assign_flat_to_params(params, flat):
    """Write flat vector back into params (in-place)."""
    offset = 0
    for p in params:
        numel = p.data.numel()
        p.data.copy_(flat[offset:offset + numel].view_as(p.data))
        offset += numel

def _add_flat_to_params(params, flat_delta, alpha=1.0):
    """p += alpha * delta, delta is flat vector."""
    offset = 0
    for p in params:
        numel = p.data.numel()
        p.data.add_(flat_delta[offset:offset + numel].view_as(p.data), alpha=alpha)
        offset += numel

# -------------------------
# ZO training
# -------------------------
def run_lora_zo(args, clip_model, logit_scale, dataset, device_id):
    """
    Zero-order (gradient-free) optimization for LoRA parameters.
    Same interface as run_lora_sgd(args, clip_model, logit_scale, dataset, device_id).
    """

    VALIDATION = True

    # apply LoRA
    list_lora_layers = apply_lora(args, clip_model)

    torch.cuda.set_device(device_id)
    clip_model = clip_model.cuda()
    mark_only_lora_as_trainable(clip_model)

    total_iters = args.n_iters * args.shots

    # ZO hyperparams (with safe defaults)
    mu = float(getattr(args, "mu", 1e-3))                 # smoothing / perturb radius
    q = int(getattr(args, "zo_samples", 2))               # number of random directions per step
    lr = float(getattr(args, "lr", 1e-3))
    weight_decay = float(getattr(args, "weight_decay", 1e-2))
    eps = 1e-12

    # cosine lr schedule (manual, because we don't use torch Optimizer.step)
    def lr_at(step_idx):
        # step_idx in [0, total_iters)
        eta_min = 1e-7
        t = min(max(step_idx, 0), total_iters)
        cos_inner = np.pi * t / float(total_iters)
        return eta_min + 0.5 * (lr - eta_min) * (1.0 + np.cos(cos_inner))

    # training logs
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_iterations = []
    learning_rates = []
    iterations = []

    template = "a photo of a {}."
    texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        texts = clip.tokenize(texts).cuda()
        class_embeddings = clip_model.encode_text(texts)
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    # cache LoRA params list
    lora_params = list(get_lora_parameters(clip_model))
    if len(lora_params) == 0:
        raise RuntimeError("No LoRA parameters found. Check apply_lora / mark_only_lora_as_trainable.")

    # main loop
    count_iters = 0
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0.0
        tot_samples = 0
        loss_epoch = 0.0

        for i, (images, target) in enumerate(dataset.train_loader):
            images, target = images.cuda(), target.cuda()

            # re-encode text features if needed (text/both)
            if args.encoder in ("text", "both"):
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            # ---------- define loss function for current params ----------
            def forward_loss():
                # vision encoder: depends on whether you allow updating vision LoRA
                if args.encoder in ("vision", "both"):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        img_emb = clip_model.encode_image(images)
                else:
                    # vision frozen: no grad needed anyway, but keep behavior aligned with your SGD code
                    with torch.no_grad():
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            img_emb = clip_model.encode_image(images)

                img_feat = img_emb / img_emb.norm(dim=-1, keepdim=True)
                logits = logit_scale * img_feat @ text_features.t()
                loss_val = F.cross_entropy(logits, target)
                return loss_val, logits

            # ---------- ZO gradient estimate on LoRA params ----------
            # save current theta
            with torch.no_grad():
                theta0 = _flatten_params(lora_params)

                # accumulate gradient estimates in flat space
                g_hat = torch.zeros_like(theta0)

                # estimate using q random directions
                for _ in range(q):
                    u = torch.randn_like(theta0)
                    u = u / (u.norm() + eps)

                    # theta+ = theta0 + mu*u
                    _assign_flat_to_params(lora_params, theta0)
                    _add_flat_to_params(lora_params, u, alpha=mu)
                    loss_pos, _ = forward_loss()
                    f_pos = float(loss_pos.detach().item())

                    # theta- = theta0 - mu*u
                    _assign_flat_to_params(lora_params, theta0)
                    _add_flat_to_params(lora_params, u, alpha=-mu)
                    loss_neg, _ = forward_loss()
                    f_neg = float(loss_neg.detach().item())

                    # two-point estimator
                    g_hat.add_(((f_pos - f_neg) / (2.0 * mu)) * u)

                g_hat.div_(float(q))

                # restore theta0
                _assign_flat_to_params(lora_params, theta0)

                # ---------- parameter update ----------
                cur_lr = lr_at(count_iters + 1)

                # weight decay (decoupled-like on params): theta <- theta * (1 - lr*wd)
                if weight_decay > 0:
                    theta0 = theta0 * (1.0 - cur_lr * weight_decay)

                # gradient step
                theta_new = theta0 - cur_lr * g_hat
                _assign_flat_to_params(lora_params, theta_new)

            # ---------- logging on current (updated) params ----------
            # compute loss/logits once (no grad needed)
            with torch.no_grad():
                loss_now, logits_now = forward_loss()
                batch_loss = float(loss_now.item())
                batch_acc = cls_acc(logits_now, target)

            acc_train += batch_acc * target.shape[0]
            loss_epoch += batch_loss * target.shape[0]
            tot_samples += target.shape[0]

            count_iters += 1
            if count_iters >= total_iters:
                break

        # epoch-ish summary (aligned with your SGD code: only record if not finished)
        if count_iters < total_iters:
            acc_train /= max(tot_samples, 1)
            loss_epoch /= max(tot_samples, 1)
            current_lr = lr_at(count_iters)

            train_losses.append(loss_epoch)
            train_accuracies.append(acc_train)
            learning_rates.append(current_lr)
            iterations.append(count_iters)

            print(f"Iter: {count_iters}, LR: {current_lr:.6f}, Acc: {acc_train:.4f}, Loss: {loss_epoch:.4f}")

        # validation
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(clip_model, dataset.val_loader, dataset.classnames)
            val_accuracies.append(acc_val)
            val_iterations.append(count_iters)
            print(f"**** Iter: {count_iters}, Val accuracy: {acc_val:.2f}. ****\n")

    plot_training_curves(
        args,
        iterations,
        train_losses,
        train_accuracies,
        val_iterations,
        val_accuracies,
        learning_rates
    )

    if args.save_path is not None:
        save_lora(args, list_lora_layers)

    return
