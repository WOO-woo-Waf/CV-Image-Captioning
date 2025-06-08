import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys
import os

# 将训练脚本所在目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from res_Qwen_ddp_v2 import VisionLanguageModel, VisualEncoder, TransformerPrefixEncoder

def load_model(model_path, lm_path, device='cuda'):
    """加载训练好的模型"""
    model = VisionLanguageModel(lm_path, freeze=True).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate_caption(model, tokenizer, image_path, device='cuda'):
    """生成图片描述"""
    # 图像预处理 - 使用与训练时相同的transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # 生成描述
    with torch.no_grad():
        # 使用模型生成caption
        vis_feat = model.visual_encoder(image)
        prefix_emb = model.prefix_encoder(vis_feat).unsqueeze(1)
        # 创建attention mask (全1，因为只有prefix embedding)
        attention_mask = torch.ones(prefix_emb.shape[0], prefix_emb.shape[1], 
                                  dtype=torch.long, device=device)
        # 准备生成参数
        generated_ids = model.language_model.generate(
            inputs_embeds=prefix_emb,
            attention_mask=attention_mask,
            max_new_tokens=20,  # 限制最大长度
            do_sample=True,
            top_k=20,           # 降低top_k值
            top_p=0.85,          # 使用nucleus sampling
            temperature=0.5,    # 适度降低温度
            repetition_penalty=2.0,  # 防止重复
            eos_token_id=tokenizer.eos_token_id,  # 明确设置结束符
            pad_token_id=tokenizer.eos_token_id   # 明确设置填充符
        )
    
    # 解码生成的文本
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption

def main():
    parser = argparse.ArgumentParser(description='Generate caption for an image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='res_Qwen_v2.pth', help='Path to the trained model')
    parser.add_argument('--lm_path', type=str, default='./model', help='Path to the language model')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for inference')
    
    args = parser.parse_args()
    
    # 检查设备可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    try:
        # 加载tokenizer和模型
        tokenizer = AutoTokenizer.from_pretrained(args.lm_path, use_fast=False)
        model = load_model(args.model_path, args.lm_path, device=args.device)
        
        # 生成描述
        caption = generate_caption(model, tokenizer, args.image_path, device=args.device)
        
        print("\nGenerated Caption:")
        print(caption)
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()