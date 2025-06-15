import os
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "true"  # 强制信任自定义代码
import torch
import json 
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
from transformers import CLIPVisionConfig
import transformers
import torch.nn.functional as F
import sys

# 视觉编码器，使用本地 CLIP 模型
class VisualEncoder(nn.Module):
    def __init__(self, model_path='/home/wangdx_lab/cse12210928/LLaVA/model/clip-vit-large-patch14-336', freeze=True):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(model_path)
        self.processor = CLIPImageProcessor.from_pretrained(model_path)
        
        if freeze:
            for p in self.clip.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = self.clip(pixel_values=x)
        return outputs.last_hidden_state  # Shape: [B, num_patches+1, hidden_size]

# QFormer，适配1024维度
class BLIPQFormer(nn.Module):
    def __init__(self, num_queries=32, d_model=1024, nhead=8, num_layers=6):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model))

        self.layers = nn.ModuleList([
            QFormerBlock(d_model, nhead)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, visual_feats):
        B = visual_feats.size(0)
        query = self.query_embed.expand(B, -1, -1)  # [B, num_queries, C]

        for layer in self.layers:
            query = layer(query, visual_feats)
        return self.norm(query)

# QFormerBlock，适配1024维度
class QFormerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, query, visual_feats):
        cross_attn_out, _ = self.cross_attn(query, visual_feats, visual_feats)
        query = self.cross_attn_norm(query + cross_attn_out)

        self_attn_out, _ = self.self_attn(query, query, query)
        query = self.self_attn_norm(query + self_attn_out)

        ffn_out = self.ffn(query)
        query = self.ffn_norm(query + ffn_out)
        return query

# PrefixTransformer，适配1024维度
class PrefixTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# VisionLanguageModel，加载本地千问2语言模型
class VisionLanguageModel(nn.Module):
    def __init__(self, lm_path='/home/wangdx_lab/cse12210928/Qwen/qwen-7b-chat/qwen/Qwen-7B-Chat', freeze=True):
        super().__init__()
        self.visual_encoder = VisualEncoder()  # 默认加载clip-vit-large-patch14-336
        self.language_model = AutoModelForCausalLM.from_pretrained(lm_path, trust_remote_code=True)
        if freeze:
            for p in self.language_model.parameters():
                p.requires_grad = False
        
        self.token_embedding = self.language_model.get_input_embeddings()
        self.embed_dim = self.token_embedding.embedding_dim
        
        self.qformer = BLIPQFormer(d_model=1024)  # 修改为1024
        self.proj = PrefixTransformer(input_dim=1024, embed_dim=self.embed_dim)

    def forward(self, images, prompt_ids, prompt_mask, cap_ids, cap_mask):
        B = images.size(0)
        device = images.device
        
        # 获取视觉特征
        vis_feat = self.visual_encoder(images)
        
        # 获取 QFormer 输出
        qformer_out = self.qformer(vis_feat)
        
        # 获取 PrefixTransformer 的嵌入
        prefix_emb = self.proj(qformer_out)  # [B, num_queries, embed_dim]
        
        # 获取语言模型的嵌入
        prompt_emb = self.token_embedding(prompt_ids)
        cap_emb = self.token_embedding(cap_ids)
        
        # 将各个嵌入拼接起来
        input_embs = torch.cat([prefix_emb, prompt_emb, cap_emb], dim=1)
        
        # 构建注意力掩码
        prefix_mask = torch.ones(B, prefix_emb.size(1), device=device, dtype=prompt_mask.dtype)
        attention_mask = torch.cat([prefix_mask, prompt_mask, cap_mask], dim=1)
        
        # 生成标签（用于计算损失）
        ignore_prefix = torch.full((B, prefix_emb.size(1)), -100, device=device, dtype=cap_ids.dtype)
        ignore_prompt = torch.full((B, prompt_ids.size(1)), -100, device=device, dtype=cap_ids.dtype)
        labels = torch.cat([ignore_prefix, ignore_prompt, cap_ids], dim=1)
        
        # 前向计算
        outputs = self.language_model(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

# COCO 数据集加载
class COCO(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.processor = CLIPImageProcessor.from_pretrained('/home/wangdx_lab/cse12210928/LLaVA/model/clip-vit-large-patch14-336')
        
         # 读取 COCO 标注文件
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 构建 image_id 到 file_name 的映射
        self.image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        self.annotations = data["annotations"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        caption = annotation['caption'].strip()  # 提取 caption 并去除多余的空格
        image_name = self.image_id_to_filename[image_id]  # 获取对应的图像文件名
        image_path = os.path.join(self.image_dir, image_name)  # 构建图像路径

        try:
            # 加载图像
            print(image_path)
            img = Image.open(image_path).convert("RGB")

            # 使用 CLIP 处理器对图像进行预处理
            if self.transform:
                img = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
            return img, caption

        except Exception as e:
            print(f"加载图片失败 idx={idx}: {e}")
            next_id = (idx + 1) % len(self.annotations)
            return self.__getitem__(next_id)


def move_to_device(batch, device='cuda'):
    return {key: value.to(device) for key, value in batch.items()}
# 训练循环
def train_single_gpu(accumulation_steps=8):
    # 读取数据
    json_path = '/home/wangdx_lab/cse12210928/LLaVA/data/coco2014/annotations/captions_train2014.json'
    image_dir = '/home/wangdx_lab/cse12210928/LLaVA/data/coco2014/train2014'

    # 初始化数据集
    dataset = COCO(json_path=json_path, image_dir=image_dir, transform=True)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    lm_path = '/home/wangdx_lab/cse12210928/Qwen/qwen-7b-chat/qwen/Qwen-7B-Chat'
    model = VisionLanguageModel(lm_path, freeze=True).cuda()  # 单卡训练
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        lm_path,
        cache_dir='./cache',
        model_max_length=256,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id
    print(tokenizer.model_max_length)
    
    optimizer = torch.optim.Adam([{'params': model.qformer.parameters()},
                                  {'params': model.proj.parameters()}], lr=1e-4, weight_decay=1e-5)

    num_epochs = 1
    scaler = GradScaler()
    for epoch in tqdm(range(num_epochs), desc="Training Epochs", dynamic_ncols=True):
        model.train()
        total_loss = 0.0

        for step, (images, captions) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training Steps", leave=False, dynamic_ncols=True):
            images = images.cuda()
            prompt_batch = ["Describe this image"] * images.size(0)

            p = tokenizer(prompt_batch, return_tensors='pt', padding='max_length',
                          truncation=True, max_length=256)
            p = move_to_device(p)

            c = tokenizer(captions, return_tensors='pt', padding='max_length',
                          truncation=True, max_length=256)
            c = move_to_device(c)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(
                    images,
                    prompt_ids=p['input_ids'],
                    prompt_mask=p['attention_mask'],
                    cap_ids=c['input_ids'],
                    cap_mask=c['attention_mask']
                )
                loss = outputs.loss

            loss = loss / accumulation_steps  # 累积梯度
            scaler.scale(loss).backward()

            # 每 `accumulation_steps` 次反向传播更新一次梯度
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()

             # 每 `display_steps` 步打印一次生成的文本
            if (step + 1) % 100 == 0:
                generated_ids = outputs.logits.argmax(dim=-1)  # 获取最大概率的词 ID
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print(f"Step {step+1} - Generated Caption: {generated_text}")
                sys.stdout.flush()

            total_loss += loss.item()

            if (step + 1) % 500 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1}, Step {step+1}/{len(dataloader)} - Avg Loss: {avg_loss:.4f}")
                print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3} GB")
                print(f"Cached memory: {torch.cuda.memory_cached() / 1024**3} GB")
                sys.stdout.flush()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'clip_qformer_2.pth')

def generate_caption_from_image(model_path, lm_path, image_path, device='cuda'):
    # 加载模型
    model = VisionLanguageModel(lm_path, freeze=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载训练好的模型权重
    model.eval()
    
    # 加载分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        lm_path,
        cache_dir='./cache',
        model_max_length=256,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    # 图像预处理
    processor = CLIPImageProcessor.from_pretrained('/home/wangdx_lab/cse12210928/LLaVA/model/clip-vit-large-patch14-336')
    img = Image.open(image_path).convert("RGB")
    img_tensor = processor(images=img, return_tensors="pt").pixel_values.squeeze(0).to(device)

    # 生成空的文本输入（让模型生成描述）
    prompt_batch = ["Describe this image"] * img_tensor.size(0)
    prompt = tokenizer(prompt_batch, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    prompt = {key: value.to(device) for key, value in prompt.items()}

     # 为生成图像描述提供填充，确保 prompt_emb 和 cap_emb 的长度一致
    max_len = 256  # 定义生成的最大长度，保持一致
    pad_prompt = F.pad(prompt['input_ids'], (0, max_len - prompt['input_ids'].size(1)), value=tokenizer.pad_token_id)
    pad_mask = F.pad(prompt['attention_mask'], (0, max_len - prompt['attention_mask'].size(1)), value=0)

    # 处理 cap_ids (同样填充到最大长度)
    cap_ids = pad_prompt  # 假设 cap_ids 为与 prompt_ids 一致，模型会根据输入生成描述
    cap_mask = pad_mask  # 假设 cap_mask 与 prompt_mask 一致

    # 获取模型的输出，并确保 prompt_emb 和 cap_emb 的长度一致
    with autocast(device_type=device):
        outputs = model(
            img_tensor.unsqueeze(0),  # 为 batch size 为 1 增加一个维度
            prompt_ids=pad_prompt,
            prompt_mask=pad_mask,
            cap_ids=cap_ids,
            cap_mask=cap_mask
        )

    # 获取生成的文本（最大概率词）
    generated_ids = outputs.logits.argmax(dim=-1)  # 获取最大概率的词 ID
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def pic(image_path):
    model_path = 'clip_qformer.pth'  # 你训练好的模型路径
    lm_path = '/home/wangdx_lab/cse12210928/Qwen/qwen-7b-chat/qwen/Qwen-7B-Chat'  # 语言模型路径

    # 使用 GPU 或 CPU 运行
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 调用生成描述函数
    caption = generate_caption_from_image(model_path, lm_path, image_path, device)
    print(f"Generated Caption: {caption}")


def load_data(json_path, image_dir, batch_size=1):
    dataset = COCO(json_path=json_path, image_dir=image_dir, transform=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader

def evaluate_model(model, dataloader, tokenizer, max_samples=1000, device='cuda'):
    model.eval()  
    total_loss = 0.0
    num_samples = 0 
    with torch.no_grad():
        for step, (images, captions) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluation Steps", leave=False, dynamic_ncols=True):
            if num_samples >= max_samples:
                break

            images = images.to(device) 

            prompt_batch = ["Describe this image"] * images.size(0)

            prompt_tokens = tokenizer(prompt_batch, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
            prompt_tokens = move_to_device(prompt_tokens, device)

            caption_tokens = tokenizer(captions, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
            caption_tokens = move_to_device(caption_tokens, device)

            with autocast(device_type='cuda'):
                outputs = model(
                    images,
                    prompt_ids=prompt_tokens['input_ids'],
                    prompt_mask=prompt_tokens['attention_mask'],
                    cap_ids=caption_tokens['input_ids'],
                    cap_mask=caption_tokens['attention_mask']
                )

                loss = outputs.loss
                total_loss += loss.item()

                generated_ids = outputs.logits.argmax(dim=-1)
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print(f"Generated Caption: {generated_text}")

            num_samples += images.size(0) 

    avg_loss = total_loss / num_samples if num_samples > 0 else 0  
    print(f"Evaluation complete. Avg Loss: {avg_loss:.4f}")
    return avg_loss

def load_model(lm_path, model_path, freeze=True):
    model = VisionLanguageModel(lm_path, freeze=freeze).cuda()
    model.load_state_dict(torch.load(model_path))  # 加载训练好的模型权重
    model.eval()
    return model

def eva():
    lm_path = '/home/wangdx_lab/cse12210928/Qwen/qwen-7b-chat/qwen/Qwen-7B-Chat'
    model_path = 'clip_qformer_2.pth'

    # 加载模型
    model = load_model(lm_path, model_path, freeze=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        lm_path,
        cache_dir='./cache',
        model_max_length=256,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    # 加载验证集
    json_path_val = '/home/wangdx_lab/cse12210928/LLaVA/data/coco2014/annotations/captions_val2014.json'
    image_dir_val = '/home/wangdx_lab/cse12210928/LLaVA/data/coco2014/val2014'
    val_dataloader = load_data(json_path_val, image_dir_val, batch_size=1)

    # 评估模型
    evaluate_model(model, val_dataloader, tokenizer)

if __name__ == "__main__":
    # image_path = '/home/wangdx_lab/cse12210928/LLaVA/data/coco2014/val2014/COCO_val2014_000000000073.jpg'  # 替换为你要测试的图片路径
    # pic(image_path)


    eva()

    # train_single_gpu()

