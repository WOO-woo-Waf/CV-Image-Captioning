import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel
from PIL import Image
import pandas as pd
from io import BytesIO
from torch.amp import autocast, GradScaler
from torchvision.models import resnet101, ResNet101_Weights

class COCO(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_data = row['image']
        caption = row['caption']
        try:
            if isinstance(img_data, dict):
                img_bytes = img_data.get('bytes', None)
                if img_bytes is None:
                    raise ValueError(f"字典内没有'image'字段，idx={idx}")
            elif isinstance(img_data, bytes):
                img_bytes = img_data
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, caption
        except Exception as e:
            print(f"加载图片失败 idx={idx}: {e}")
            next_id = (idx + 1) % len(self.df)
            return self.__getitem__(next_id)
        

class VisualEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        resnet = resnet101(ResNet101_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
        
        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        
        self.proj = nn.Sequential(
            nn.Linear(1024, 1536),
            nn.GELU(),
            nn.LayerNorm(1536),
            nn.Linear(1536, 768),
            nn.GELU(),
            nn.LayerNorm(768)
        )
        
        self.position_emb = nn.Parameter(torch.randn(1, 16, 768) * 0.02) # 这里加上一个位置编码，不过可以考虑不要
    
    def forward(self, x):
        x = self.features(x)  # (B, 2048, 7, 7)
        x = self.spatial_reduce(x)  # (B, 1024, 4, 4)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, 16, 1024)
        x = self.proj(x)  # (B, 16, 768)
        return x + self.position_emb  


class BLIPQFormer(nn.Module):
    def __init__(self, num_queries=32, d_model=768, nhead=8, num_layers=6):
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

class VisionLanguageModel(nn.Module):
    def __init__(self, lm_path, freeze=True):
        super().__init__()
        self.visual_encoder = VisualEncoder()
        self.language_model = AutoModelForCausalLM.from_pretrained(lm_path)
        self.language_model.gradient_checkpointing_enable()
        if freeze:
            for p in self.language_model.parameters():
                p.requires_grad = False
        
        self.token_embedding = self.language_model.get_input_embeddings()
        self.embed_dim = self.token_embedding.embedding_dim
        
        self.qformer = BLIPQFormer(d_model=768)
        
        self.proj = PrefixTransformer(input_dim=768, embed_dim=self.embed_dim)

    def forward(self, images, prompt_ids, prompt_mask, cap_ids, cap_mask):
        B = images.size(0)
        device = images.device
        
        vis_feat = self.visual_encoder(images)
        
        qformer_out = self.qformer(vis_feat)
        
        prefix_emb = self.proj(qformer_out)  # [B, num_queries, embed_dim]
        
        prompt_emb = self.token_embedding(prompt_ids)
        cap_emb = self.token_embedding(cap_ids)
        
        input_embs = torch.cat([prefix_emb, prompt_emb, cap_emb], dim=1)
        
        prefix_mask = torch.ones(B, prefix_emb.size(1), device=device, dtype=prompt_mask.dtype)
        attention_mask = torch.cat([prefix_mask, prompt_mask, cap_mask], dim=1)
        
        ignore_prefix = torch.full((B, prefix_emb.size(1)), -100, device=device, dtype=cap_ids.dtype)
        ignore_prompt = torch.full((B, prompt_ids.size(1)), -100, device=device, dtype=cap_ids.dtype)
        labels = torch.cat([ignore_prefix, ignore_prompt, cap_ids], dim=1)
        
        outputs = self.language_model(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

def setup_distributed():
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def train_ddp(rank, world_size):
    data_folder = './data/data'
    all_data = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.parquet') and file.startswith('train'):
                df = pd.read_parquet(os.path.join(root, file))
                all_data.append(df)
    combined_data = pd.concat(all_data, ignore_index=True)
    transform = ResNet101_Weights.IMAGENET1K_V2.transforms()
    dataset = COCO(combined_data, transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=4)

    lm_path = './model'
    model = VisionLanguageModel(lm_path, freeze=True).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    tokenizer = AutoTokenizer.from_pretrained(lm_path, use_fast=False)
    
    optimizer = torch.optim.Adam([
        {'params': model.module.qformer.parameters()},
        {'params': model.module.proj.parameters()},
        {'params': model.module.visual_encoder.proj.parameters()},
        {'params': model.module.visual_encoder.spatial_reduce.parameters()},
        {'params': [model.module.visual_encoder.position_emb]}
    ], lr=1e-4, weight_decay=1e-5)

    # 因为只能跑两天，所以一个epoch，一个epoch的来跑
    checkpoint_path = 'clip_qformer.pth'
    if rank == 0 and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if rank == 0:
            print(f"恢复训练")

    num_epochs = 3
    scaler = GradScaler()
    prompt_text = "Describe the image:"
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for step, (images, captions) in enumerate(dataloader):
            images = images.to(rank)
            prompt_batch = [prompt_text] * images.size(0)  # 现在是没加提示词，其实也可以把提示词加上去
            p = tokenizer(prompt_batch, return_tensors='pt', padding='max_length',
                          truncation=True, max_length=60).to(rank)
            c = tokenizer(captions, return_tensors='pt', padding='max_length',
                          truncation=True, max_length=60).to(rank)

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

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if rank == 0 and (step + 1) % 500 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"[Rank {rank}] Epoch {epoch+1}, Step {step+1}/{len(dataloader)} - Avg Loss: {avg_loss:.4f}")

        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")

    if rank == 0:
        # torch.save(model.module.state_dict(), 'clip_qformer.pth')
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

if __name__ == "__main__":
    rank, world_size = setup_distributed()
    train_ddp(rank, world_size)
    cleanup_distributed()