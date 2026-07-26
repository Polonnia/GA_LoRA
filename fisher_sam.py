import time
import torch
import torch.nn.functional as F

from utils import *
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, save_lora


class FSAM(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        base_optimizer,
        rho=0.05,
        adaptive=False,
        keep_ratio=0.1,
        mask_update_interval=100,
        fisher_beta=0.9,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert 0.0 < keep_ratio <= 1.0, f"keep_ratio must be in (0, 1], got {keep_ratio}"
        assert mask_update_interval >= 1, "mask_update_interval must be >= 1"
        assert 0.0 <= fisher_beta < 1.0, f"fisher_beta must be in [0, 1), got {fisher_beta}"

        defaults = dict(
            rho=rho,
            adaptive=adaptive,
            keep_ratio=keep_ratio,
            mask_update_interval=mask_update_interval,
            fisher_beta=fisher_beta,
            **kwargs,
        )
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.global_step = 0

    @torch.no_grad()
    def _update_fisher_and_mask(self):
        fisher_values = []
        fisher_refs = []

        for group in self.param_groups:
            keep_ratio = group["keep_ratio"]
            fisher_beta = group["fisher_beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if "fisher" not in state:
                    state["fisher"] = torch.zeros_like(p)
                if "mask" not in state:
                    state["mask"] = torch.ones_like(p)

                fisher = state["fisher"]
                fisher.mul_(fisher_beta).add_((1.0 - fisher_beta) * (p.grad.detach() ** 2))

                fisher_values.append(fisher.view(-1))
                fisher_refs.append((p, fisher.numel(), keep_ratio))

        if not fisher_values:
            return

        if self.global_step % self.param_groups[0]["mask_update_interval"] != 0:
            return

        all_scores = torch.cat(fisher_values, dim=0)
        total_params = all_scores.numel()
        keep_ratio = self.param_groups[0]["keep_ratio"]
        keep_count = max(1, int(total_params * keep_ratio))

        if keep_count >= total_params:
            threshold = None
        else:
            topk_vals, _ = torch.topk(all_scores, k=keep_count, largest=True, sorted=False)
            threshold = topk_vals.min()

        start = 0
        for p, n, _ in fisher_refs:
            fisher = self.state[p]["fisher"]
            if threshold is None:
                self.state[p]["mask"] = torch.ones_like(fisher)
            else:
                self.state[p]["mask"] = (fisher >= threshold).to(fisher.dtype)
            start += n

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.global_step += 1
        self._update_fisher_and_mask()

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            adaptive = group["adaptive"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                state["old_p"] = p.data.clone()
                if "mask" not in state:
                    state["mask"] = torch.ones_like(p)

                masked_grad = p.grad * state["mask"]
                e_w = ((torch.pow(p, 2) if adaptive else 1.0) * masked_grad) * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "FSAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                mask = self.state[p].get("mask", torch.ones_like(p))
                grad = p.grad * mask
                scaled = (torch.abs(p) if adaptive else 1.0) * grad
                norms.append(scaled.norm(p=2).to(shared_device))

        if not norms:
            return torch.tensor(0.0, device=shared_device)

        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def run_lora_fisher_sam(args, clip_model, logit_scale, dataset, device_id):
    VALIDATION = True

    import clip

    list_lora_layers = apply_lora(args, clip_model)

    torch.cuda.set_device(device_id)
    clip_model = clip_model.cuda()

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    base_optimizer = torch.optim.AdamW
    optimizer = FSAM(
        get_lora_parameters(clip_model),
        base_optimizer,
        rho=float(getattr(args, "rho", 0.05)),
        adaptive=bool(getattr(args, "adaptive_sam", False)),
        keep_ratio=float(getattr(args, "fsam_keep_ratio", 0.1)),
        mask_update_interval=int(getattr(args, "fsam_mask_update_interval", 100)),
        fisher_beta=float(getattr(args, "fsam_fisher_beta", 0.9)),
        lr=float(getattr(args, "lr", 1e-3)),
        betas=getattr(args, "betas", (0.9, 0.999)),
        eps=float(getattr(args, "eps", 1e-8)),
        weight_decay=float(getattr(args, "weight_decay", 0.01)),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        total_iters,
        eta_min=1e-7,
    )

    use_amp = bool(getattr(args, "use_amp", False))

    count_iters = 0

    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_iterations = []
    learning_rates = []
    iterations = []

    template = "a photo of a {}."
    texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
    texts = clip.tokenize(texts).cuda()

    with torch.no_grad():
        clip_model.eval()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            class_embeddings = clip_model.encode_text(texts)
        text_features_static = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    start_time = time.time()
    while count_iters < total_iters:
        clip_model.train()

        acc_train = 0.0
        tot_samples = 0
        loss_epoch = 0.0

        for _, (images, target) in enumerate(dataset.train_loader):
            images, target = images.cuda(), target.cuda()

            if args.encoder in ["text", "both"]:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            else:
                text_features = text_features_static

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                image_embeddings = clip_model.encode_image(images)
                image_features = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

                tf = text_features.to(device=image_features.device, dtype=image_features.dtype)
                ls = logit_scale
                if isinstance(ls, torch.Tensor):
                    ls = ls.to(device=image_features.device, dtype=image_features.dtype)

                logits = ls * (image_features @ tf.t())
                loss = F.cross_entropy(logits.float(), target)

            with torch.no_grad():
                acc_train += cls_acc(logits, target) * target.shape[0]
                loss_epoch += float(loss.item()) * target.shape[0]
                tot_samples += target.shape[0]

            loss.backward()
            optimizer.first_step(zero_grad=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                image_embeddings2 = clip_model.encode_image(images)
                image_features2 = image_embeddings2 / image_embeddings2.norm(dim=-1, keepdim=True)

                tf2 = text_features.to(device=image_features2.device, dtype=image_features2.dtype)
                ls2 = logit_scale
                if isinstance(ls2, torch.Tensor):
                    ls2 = ls2.to(device=image_features2.device, dtype=image_features2.dtype)

                logits2 = ls2 * (image_features2 @ tf2.t())
                loss2 = F.cross_entropy(logits2.float(), target)

            loss2.backward()
            optimizer.second_step(zero_grad=True)
            scheduler.step()

            count_iters += 1

            if count_iters % 10 == 0:
                print(
                    f"Iter: {count_iters}/{total_iters}, "
                    f"Loss: {float(loss.item()):.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

            if count_iters >= total_iters:
                break

        if tot_samples > 0:
            acc_train_epoch = acc_train / tot_samples
            loss_epoch_avg = loss_epoch / tot_samples
            current_lr = scheduler.get_last_lr()[0]

            train_losses.append(loss_epoch_avg)
            train_accuracies.append(acc_train_epoch)
            learning_rates.append(current_lr)
            iterations.append(count_iters)

            print(
                "Iter: {}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}".format(
                    count_iters, current_lr, acc_train_epoch, loss_epoch_avg
                )
            )

        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(clip_model, dataset.val_loader, dataset.classnames)
            val_accuracies.append(acc_val)
            val_iterations.append(count_iters)
            print("**** Iter: {}, Val accuracy: {:.2f}. ****\n".format(count_iters, acc_val))

        if count_iters >= total_iters:
            break

    end_time = time.time()
    print(f"FSAM training completed in {(end_time - start_time)/60:.2f} minutes.")

    plot_training_curves(
        args,
        iterations,
        train_losses,
        train_accuracies,
        val_iterations,
        val_accuracies,
        learning_rates,
    )

    if args.save_path is not None:
        save_lora(args, list_lora_layers)

    return
