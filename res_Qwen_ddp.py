import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from io import BytesIO


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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
        except Exception as e:
            print(f"加载图片失败 idx={idx}: {e}")
            img = torch.zeros(3, 224, 224)
        return img, caption

class VisualEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        resnet = resnet50(ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class MLPConnector(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=1536, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

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
        self.mlp = MLPConnector(output_dim=self.embed_dim)
    
    def forward(self, images, prompt_ids, prompt_mask, cap_ids, cap_mask):
        B = images.size(0)
        device = images.device
        vis_feat = self.visual_encoder(images)
        prefix_emb = self.mlp(vis_feat).unsqueeze(1)
        prefix_mask = torch.ones(B, 1, device=device, dtype=prompt_mask.dtype)

        prompt_emb = self.token_embedding(prompt_ids)
        cap_emb = self.token_embedding(cap_ids)

        input_embs = torch.cat([prefix_emb, prompt_emb, cap_emb], dim=1)
        attention_mask = torch.cat([prefix_mask, prompt_mask, cap_mask], dim=1)
        ignore = torch.full((B, 1 + prompt_ids.size(1)), -100, device=device, dtype=cap_ids.dtype)
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

def train_ddp(rank, world_size):

    data_folder = './data/data'
    all_data = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.parquet') and file.startswith('train'):
                df = pd.read_parquet(os.path.join(root, file))
                all_data.append(df)
    combined_data = pd.concat(all_data, ignore_index=True)

    dataset = COCO(combined_data, transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=4)

    lm_path = './model'
    model = VisionLanguageModel(lm_path, freeze=True).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    tokenizer = AutoTokenizer.from_pretrained(lm_path, use_fast=False)

    optimizer = torch.optim.Adam(model.module.mlp.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 3
    prompt_text = "Describe the image:"
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for step, (images, captions) in enumerate(dataloader):
            images = images.to(rank)
            prompt_batch = [prompt_text] * images.size(0)
            p = tokenizer(prompt_batch, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=100).to(rank)
            c = tokenizer(captions, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=100).to(rank)

            outputs = model(
                images,
                prompt_ids=p['input_ids'],
                prompt_mask=p['attention_mask'],
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
    
    if rank == 0:
        torch.save(model.module.state_dict(), 'res_Qwen.pth')

if __name__ == "__main__":
    rank, world_size = setup_distributed()
    train_ddp(rank, world_size)
    cleanup_distributed()