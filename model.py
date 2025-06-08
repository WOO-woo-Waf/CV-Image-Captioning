import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = load_dataset('jxie/coco_captions', split='train')
val_dataset = load_dataset('jxie/coco_captions', split='validation')

def preprocess_data(example):
    image = Image.open(example["filename"]).convert("RGB")  # 替换为实际图像路径
    image = transform(image)
    # caption = example["caption"][0] if isinstance(example["caption"], list) else example["caption"]
    caption = example['caption']
    return {"image": image, "caption": caption}

dataset = dataset.map(preprocess_data)
val_dataset = val_dataset.map(preprocess_data)

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    return {"images": images, "captions": captions}

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

class VisualEncoder(nn.Module):
    def __init__(self, freeze=True):
        super(VisualEncoder, self).__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        if freeze:
            for p in self.features.parameters(): p.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1) # [B, C, 1, 1] -> [B, C*1*1] 调整成符合语言模型的输入

class MLPConnector(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=2048):
        super(MLPConnector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class VisionLanguageModel(nn.Module):
    def __init__(self, freeze=True, lm='Qwen/Qwen2-0.5B'):
        super(VisionLanguageModel, self).__init__()
        self.visual_encoder = VisualEncoder()
        self.mlp = MLPConnector()
        self.language_model = AutoModelForCausalLM.from_pretrained(lm)
        if freeze:
            for p in self.language_model.parameters(): p.requires_grad = False
        self.token_embedding = self.language_model.get_input_embeddings()
    
    def forward(self, images, input_ids=None, labels=None):
        visual_features = self.visual_encoder(images)
        mlp_features = self.mlp(visual_features)
        input_embeds = mlp_features.unsqueeze(1)
        
        if input_ids is not None:
            token_embedding = self.token_embedding(input_ids)
            input_embeds = torch.cat([input_embeds, token_embedding], dim=1)
        
        outputs = self.language_model(inputs_embeds=input_embeds, labels=labels)
        return outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionLanguageModel().to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

optimizer = torch.optim.Adam(model.mlp.parameters(), lr=1e-4, weight_decay=1e-5)
num_epochs = 10
prompt = "Describe the image:" 

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        images = batch['images'].to(device)
        captions = batch['captions']
        
        prompt_inputs = tokenizer([prompt] * len(captions), return_tensors='pt', padding=True, max_length=100).to(device) # 这里面len(captions)==batch_size
        input_ids = prompt_inputs['input_ids']  # [batch_size, prompt_len]
        
        target_inputs = tokenizer(captions, return_tensors='pt', padding=True, max_length=100).to(device)
        target_ids = target_inputs['input_ids']  # [batch_size, target_len]
        
        # 调整 labels，忽略视觉特征和提示词部分
        batch_size = images.size(0)
        prompt_len = input_ids.size(1)
        labels = torch.cat([
            torch.full((batch_size, 1 + prompt_len), -100, device=device), 
            target_ids
        ], dim=1)  # 生成一个全是-100的张量，形状是[B, 1 + prompt_len], 只要文字部分来计算loss
        
        outputs = model(images, input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_dataloader)}")

torch.save(model, "model.pth")

