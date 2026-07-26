import time
import torch
import torch.nn.functional as F
import numpy as np

from utils import *
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, save_lora


class FocalSAM(torch.optim.Optimizer):
    """
    Focal Sharpness-Aware Minimization (Focal-SAM) Optimizer
    
    Assigns different penalties to class-wise sharpness, achieving fine-grained 
    control without extra backpropagations, while maintaining efficiency.
    
    Paper: "Focal-SAM: Focal Sharpness-Aware Minimization for Long-Tailed Classification"
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FocalSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step: Store gradients and compute perturbation direction
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                self.state[p]["grad"] = p.grad.clone()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step: Apply perturbation to reach the local maximum
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def third_step(self, zero_grad=False):
        """
        Third step: Restore original weights and apply the base optimizer
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue     
                
                p.data = self.state[p]["old_p"]
                p.grad.add_(self.state[p]["grad"])

        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Single step with closure (for compatibility)
        """
        assert closure is not None, "FocalSAM requires closure"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)
        closure()
        self.third_step(zero_grad=True)

    def _grad_norm(self):
        """
        Calculate gradient norm
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def run_lora_focal_sam(args, clip_model, logit_scale, dataset, device_id):
    """
    Train CLIP model with LoRA using Focal-SAM optimizer
    
    Args:
        args: Configuration arguments
        clip_model: CLIP model to train
        logit_scale: Temperature scale for logits
        dataset: Training/validation dataset
        device_id: GPU device ID
    """
    VALIDATION = True

    import clip

    list_lora_layers = apply_lora(args, clip_model)

    torch.cuda.set_device(device_id)
    clip_model = clip_model.cuda()

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    # ========= Focal-SAM + AdamW =========
    base_optimizer = torch.optim.AdamW
    optimizer = FocalSAM(
        get_lora_parameters(clip_model),
        base_optimizer,
        rho=float(getattr(args, "rho", 0.05)),
        adaptive=bool(getattr(args, "adaptive_sam", False)),
        lr=float(getattr(args, "lr", 1e-3)),
        betas=getattr(args, "betas", (0.9, 0.999)),
        eps=float(getattr(args, "eps", 1e-8)),
        weight_decay=float(getattr(args, "weight_decay", 0.01)),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        total_iters,
        eta_min=1e-7
    )

    # training LoRA
    use_amp = bool(getattr(args, "use_amp", False))

    count_iters = 0

    # 记录训练过程（保留 plot 所需）
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_iterations = []
    learning_rates = []
    iterations = []

    # ====== 文本特征（encoder=vision 推荐固定）======
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

        for i, (images, target) in enumerate(dataset.train_loader):
            images, target = images.cuda(), target.cuda()

            # encoder=vision: text fixed; text/both: 每步可更新
            if args.encoder in ['text', 'both']:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            else:
                text_features = text_features_static

            # -------------------------
            # 1st forward-backward (compute loss and gradients)
            # -------------------------
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

            # 统计用第一次 loss/logits
            with torch.no_grad():
                acc_train += cls_acc(logits, target) * target.shape[0]
                loss_epoch += float(loss.item()) * target.shape[0]
                tot_samples += target.shape[0]

            loss.backward()
            optimizer.first_step(zero_grad=True)

            # -------------------------
            # 2nd forward-backward (at w+e(w), SAM step)
            # -------------------------
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

            # -------------------------
            # 3rd forward-backward (final update)
            # -------------------------
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                image_embeddings3 = clip_model.encode_image(images)
                image_features3 = image_embeddings3 / image_embeddings3.norm(dim=-1, keepdim=True)

                tf3 = text_features.to(device=image_features3.device, dtype=image_features3.dtype)
                ls3 = logit_scale
                if isinstance(ls3, torch.Tensor):
                    ls3 = ls3.to(device=image_features3.device, dtype=image_features3.dtype)

                logits3 = ls3 * (image_features3 @ tf3.t())
                loss3 = F.cross_entropy(logits3.float(), target)

            loss3.backward()
            optimizer.third_step(zero_grad=True)
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

        # ===== epoch/阶段结束：写入画图数据 =====
        if tot_samples > 0:
            acc_train_epoch = acc_train / tot_samples
            loss_epoch_avg = loss_epoch / tot_samples
            current_lr = scheduler.get_last_lr()[0]

            train_losses.append(loss_epoch_avg)
            train_accuracies.append(acc_train_epoch)
            learning_rates.append(current_lr)
            iterations.append(count_iters)

            print('Iter: {}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(
                count_iters, current_lr, acc_train_epoch, loss_epoch_avg))

        # ===== validation（保留画图）=====
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(clip_model, dataset.val_loader, dataset.classnames)
            val_accuracies.append(acc_val)
            val_iterations.append(count_iters)
            print("**** Iter: {}, Val accuracy: {:.2f}. ****\n".format(count_iters, acc_val))

        if count_iters >= total_iters:
            break

    end_time = time.time()
    print(f"Focal-SAM training completed in {(end_time - start_time)/60:.2f} minutes.")
    
    # ===== plot 不能丢 =====
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
