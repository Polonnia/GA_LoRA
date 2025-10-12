from tqdm import tqdm
import torch
import clip

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            # Use single template (first entry) for zero-shot classifier
            text = template[0].format(classname)
            texts = clip.tokenize([text]).cuda()
            # Handle DataParallel wrapper
            model = clip_model.module if hasattr(clip_model, 'module') else clip_model
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings[0]
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
    return clip_weights


def pre_load_features(clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())
        features, labels = torch.cat(features), torch.cat(labels)
    
    return features, labels

def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

