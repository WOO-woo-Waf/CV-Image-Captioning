import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import resnet101, ResNet101_Weights
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from io import BytesIO

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
            new_idx = (idx + 1) % len(self.df)
            return self.__getitem__(new_idx)

class VisualEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        resnet = resnet101(ResNet101_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # (B, 2048, 7, 7)
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)                    # (B, 2048, 7, 7)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)   # (B, 49, 2048)
        return x

class TransformerPrefixEncoder(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=1536, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: (B, 49, 2048)
        x = self.proj(x)         # (B, 49, D)
        x = self.encoder(x)      # (B, 49, D)
        return self.norm(x)      # (B, 49, D)

class VisionLanguageModel(nn.Module):
    def __init__(self, lm_path, freeze=True):
        super().__init__()
        self.visual_encoder = VisualEncoder()
        self.language_model = AutoModelForCausalLM.from_pretrained(lm_path)
        if freeze:
            for p in self.language_model.parameters():
                p.requires_grad = False
        self.token_embedding = self.language_model.get_input_embeddings()
        self.embed_dim = self.token_embedding.embedding_dim
        self.prefix_encoder = TransformerPrefixEncoder(input_dim=2048, embed_dim=self.embed_dim)

    def forward(self, images, cap_ids, cap_mask):
        B = images.size(0)
        device = images.device

        vis_feat = self.visual_encoder(images)           # (B, 49, 2048)
        prefix_emb = self.prefix_encoder(vis_feat)       # (B, 49, D)
        cap_emb = self.token_embedding(cap_ids)          # (B, L, D)

        input_embs = torch.cat([prefix_emb, cap_emb], dim=1)  # (B, 49+L, D)

        prefix_mask = torch.ones(B, prefix_emb.size(1), device=device, dtype=cap_mask.dtype)
        attention_mask = torch.cat([prefix_mask, cap_mask], dim=1)

        ignore = torch.full((B, prefix_emb.size(1)), -100, device=device, dtype=cap_ids.dtype)
        labels = torch.cat([ignore, cap_ids], dim=1)

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

@torch.no_grad()
def evaluate(model, val_dataloader, tokenizer, rank):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for images, captions in val_dataloader:
        images = images.to(rank)
        c = tokenizer(captions, return_tensors='pt', padding='max_length', truncation=True, max_length=100).to(rank)
        outputs = model(images, cap_ids=c['input_ids'], cap_mask=c['attention_mask'])
        total_loss += outputs.loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    if rank == 0:
        print(f"[Validation] loss: {avg_loss:.4f}")
    return avg_loss


def train_ddp(rank, world_size):
    data_folder = './data/data'
    train = []
    val = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.parquet') :
                if file.startswith('train'):
                    df = pd.read_parquet(os.path.join(root, file))
                    train.append(df)
                elif file.startswith('val'):
                    df = pd.read_parquet(os.path.join(root, file))
                    val.append(df)
    train_data = pd.concat(train, ignore_index=True)
    val_data = pd.concat(val, ignore_index=True)

    transform = ResNet101_Weights.IMAGENET1K_V2.transforms()

    train_dataset = COCO(train_data, transform)
    val_dataset = COCO(val_data, transform)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(train_dataset, batch_size=2, sampler=sampler, num_workers=4)
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, sampler=val_sampler, num_workers=4)

    lm_path = './model'
    model = VisionLanguageModel(lm_path, freeze=True).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    tokenizer = AutoTokenizer.from_pretrained(lm_path, use_fast=False)

    optimizer = torch.optim.Adam(model.module.prefix_encoder.parameters(), lr=1e-4, weight_decay=1e-5)

    val_loss = float('inf')
    num_epochs = 10
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for step, (images, captions) in enumerate(dataloader):
            images = images.to(rank)
            c = tokenizer(captions, return_tensors='pt', padding='max_length',
                          truncation=True, max_length=100).to(rank)

            outputs = model(
                images,
                cap_ids=c['input_ids'],
                cap_mask=c['attention_mask']
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if rank == 0 and (step + 1) % 500 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"[Rank {rank}] Epoch {epoch+1}, Step {step+1}/{len(dataloader)} - Avg Loss: {avg_loss:.4f}")

        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")
            temp = evaluate(model, val_dataloader, tokenizer, rank)   
            val_path = "res_Qwen_val.pth"
            if temp < val_loss:
                val_loss = temp
                torch.save(model.module.state_dict(), val_path)

    
    if rank == 0:
        torch.save(model.module.state_dict(), 'res_Qwen_v2.pth')

if __name__ == "__main__":
    rank, world_size = setup_distributed()
    train_ddp(rank, world_size)
    cleanup_distributed()
