import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Optimizer
from copy import deepcopy
from tqdm import tqdm

# 假设这些是你项目中现有的工具库，请根据实际情况保留或调整导入路径
from utils import * 
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
import clip  # 假设使用了 openai/CLIP 或 open_clip

# ==========================================
# Part 1: EntropySGD Optimizer Class
# 来源: ucla-vision/entropy-sgd/python/optim.py
# ==========================================
class EntropySGD(Optimizer):
    def __init__(self, params, config = {}):
        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        # EntropySGD 强制要求传入 closure, model 和 criterion
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for Entropy-SGD, model and criterion'
        
        # 初始计算一次 Loss 和 Error
        mf, merr = closure()

        c = self.config
        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = int(c['L'])
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']

        params = self.param_groups[0]['params']

        state = self.state
        # 初始化状态
        if not 't' in state:
            state['t'] = 0
            state['wc'], state['mdw'] = [], []
            for w in params:
                state['wc'].append(deepcopy(w.data))
                state['mdw'].append(deepcopy(w.grad.data))

            state['langevin'] = dict(mw=deepcopy(state['wc']),
                                    mdw=deepcopy(state['mdw']),
                                    eta=deepcopy(state['mdw']),
                                    lr = 0.1,
                                    beta1 = 0.75)

        lp = state['langevin']
        for i, w in enumerate(params):
            state['wc'][i].copy_(w.data)
            lp['mw'][i].copy_(w.data)
            lp['mdw'][i].zero_()
            lp['eta'][i].normal_()

        state['debug'] = dict(wwpd=0, df=0, dF=0, g=0, eta=0)
        llr, beta1 = lp['lr'], lp['beta1']
        g = g0 * (1 + g1) ** state['t']

        # 内部 Langevin 循环 (L次)
        for i in range(L):
            # 每次迭代都需要重新计算梯度，因此调用 closure
            f, _ = closure()
            for wc, w, mw, mdw, eta in zip(state['wc'], params, \
                                    lp['mw'], lp['mdw'], lp['eta']):
                dw = w.grad.data

                if wd > 0:
                    dw.add_(wd, w.data)
                if mom > 0:
                    mdw.mul_(mom).add_(1-damp, dw)
                    if nesterov:
                        dw.add_(mom, mdw)
                    else:
                        dw = mdw

                # 添加噪声 (Langevin Dynamics)
                eta.normal_()
                dw.add_(-g, wc - w.data).add_(eps / np.sqrt(0.5 * llr), eta)

                # 更新权重
                w.data.add_(-llr, dw)
                mw.mul_(beta1).add_(1 - beta1, w.data)

        if L > 0:
            # 将模型权重恢复到 Langevin 采样后的平均值或相关状态，并准备最终梯度
            for i, w in enumerate(params):
                w.data.copy_(state['wc'][i])
                w.grad.data.copy_(w.data - lp['mw'][i])

        # 最后使用外层 SGD 更新一次参数
        for w, mdw, mw in zip(params, state['mdw'], lp['mw']):
            dw = w.grad.data

            if wd > 0:
                dw.add_(wd, w.data)
            if mom > 0:
                mdw.mul_(mom).add_(1-damp, dw)
                if nesterov:
                    dw.add_(mom, mdw)
                else:
                    dw = mdw

            w.data.add_(-lr, dw)
        
        state['t'] += 1
        return mf, merr

# ==========================================
# Part 2: Modified Training Function
# ==========================================

def run_lora_entropy_sgd(args, clip_model, logit_scale, dataset, device_id):
    """
    Args:
        use_entropy_sgd (bool): 是否使用 EntropySGD 优化器。默认为 True。
    """
    VALIDATION = True
  
    list_lora_layers = apply_lora(args, clip_model)

    torch.cuda.set_device(device_id)
    clip_model = clip_model.cuda()
    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    # 1. 初始化优化器
    print("Using Entropy-SGD Optimizer.")
    # 配置参数参考自论文实现
    optimizer = EntropySGD(
        get_lora_parameters(clip_model),
        config=dict(
            lr=args.lr if hasattr(args, 'lr') else 0.01, # 建议默认学习率
            momentum=0.9, 
            nesterov=True, 
            weight_decay=1e-2,
            L=20,       # Langevin 迭代次数，论文推荐 20
            eps=1e-4,   # 噪声因子 (noise)
            g0=1e-4,    # gamma (scope)
            g1=0        # scoping
        )
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        total_iters,
        eta_min=1e-7
    )
    
    # 注意：EntropySGD 内部直接操作 w.grad.data，与 GradScaler 的梯度缩放不兼容。
    scaler = None

    count_iters = 0
    
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_iterations = []
    learning_rates = []
    iterations = []
    
    template = "a photo of a {}."
    texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
    
    # 预计算文本特征 (如果文本编码器不参与训练)
    with torch.no_grad():
        # 兼容不同版本的 clip 库调用
        tokenized_texts = clip.tokenize(texts).cuda()
        class_embeddings = clip_model.encode_text(tokenized_texts)
    text_features_fixed = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.

        for i, (images, target) in enumerate(dataset.train_loader):
            images, target = images.cuda(), target.cuda()
            batch_size = target.shape[0]

            # =======================================================
            # 定义 Closure (闭包)
            # EntropySGD 需要多次调用此函数来计算梯度和 Loss
            # =======================================================
            def closure():
                optimizer.zero_grad()
                
                # 前向传播 (Forward)
                # 根据你的设置决定是否重新计算文本特征
                if getattr(args, 'encoder', 'both') in ['text', 'both']:
                    # 如果训练文本端，则需重新编码
                    curr_embeddings = clip_model.encode_text(tokenized_texts)
                    curr_text_features = curr_embeddings / curr_embeddings.norm(dim=-1, keepdim=True)
                else:
                    curr_text_features = text_features_fixed

                # 视觉端特征
                # 注意：EntropySGD 需要在每次 Langevin 步中都有梯度，
                # 所以确保 encode_image 在计算图中
                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 计算 Loss
                cosine_similarity = logit_scale * image_features @ curr_text_features.t()
                loss = F.cross_entropy(cosine_similarity, target)
                
                # 反向传播 (Backward)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 计算准确率 (用于统计)
                # cls_acc 是你 utils 中的函数
                acc = cls_acc(cosine_similarity, target) * batch_size
                
                # EntropySGD 要求 closure 返回 (loss_value, metric_value)
                return loss.item(), acc

            # =======================================================
            # 执行 Optimizer Step
            # =======================================================
            # EntropySGD step 必须接收 closure
            # 虽然 model 和 criterion 参数在 python 版实现中可能未直接使用，但为了通过 assert 检查必须传入
            loss_val, acc_val = optimizer.step(closure, clip_model, F.cross_entropy)

            # 记录数据
            acc_train += acc_val
            loss_epoch += loss_val * batch_size
            tot_samples += batch_size
            
            scheduler.step()
            count_iters += 1
            
            if count_iters == total_iters:
                break
            
        # 每个 Epoch 结束后的统计
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            
            train_losses.append(loss_epoch)
            train_accuracies.append(acc_train)
            learning_rates.append(current_lr)
            iterations.append(count_iters)
            
            print('Iter: {}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(
                count_iters, current_lr, acc_train, loss_epoch))

        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(clip_model, dataset.val_loader, dataset.classnames)
            val_accuracies.append(acc_val)
            val_iterations.append(count_iters)
            print("**** Iter: {}, Val accuracy: {:.2f}. ****\n".format(count_iters, acc_val))
    
    plot_training_curves(
        args,
        iterations,
        train_losses,
        train_accuracies,
        val_iterations,
        val_accuracies,
        learning_rates
    )
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return