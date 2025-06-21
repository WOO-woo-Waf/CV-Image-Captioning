import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from torch.amp import autocast
import argparse
import torch.nn.functional as F
import gc

from res_Qwen_7B import VisionLanguageModel  # 模型定义

# 参数设置
parser = argparse.ArgumentParser(description='Generate caption for an image')
parser.add_argument('--image_path', type=str, default='sample_11129.jpg', help='Path to the input image')
parser.add_argument('--model_path', type=str, default='./model', help='Path to the language model')
parser.add_argument('--checkpoint_path', type=str, default='clip_qformer.pth', help='Trained checkpoint path')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
args = parser.parse_args()


def load_model_and_tokenizer(model_path, checkpoint_path, device):
    print(">> Loading model...")
    model = VisionLanguageModel(model_path, freeze=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir='./cache',
        model_max_length=128,  # 降低token长度
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id if hasattr(tokenizer, 'eod_id') else tokenizer.eos_token_id
    return model, tokenizer


def preprocess_image(image_path, device, target_size=(224, 224)):
    print(">> Processing image...")
    processor = CLIPImageProcessor.from_pretrained('/Clip')
    img = Image.open(image_path).convert("RGB").resize(target_size)
    img_tensor = processor(images=img, return_tensors="pt").pixel_values.squeeze(0).unsqueeze(0).to(device)
    return img_tensor


def generate_caption(model, tokenizer, image_tensor, device, max_len=128):
    prompt_text = "Describe this image"
    prompt_batch = [prompt_text]

    prompt = tokenizer(prompt_batch, return_tensors='pt', padding='max_length',
                       truncation=True, max_length=max_len)
    prompt = {k: v.to(device) for k, v in prompt.items()}

    pad_prompt = F.pad(prompt['input_ids'], (0, max_len - prompt['input_ids'].size(1)), value=tokenizer.pad_token_id)
    pad_mask = F.pad(prompt['attention_mask'], (0, max_len - prompt['attention_mask'].size(1)), value=0)
    cap_ids = pad_prompt
    cap_mask = pad_mask

    with torch.no_grad():
        with autocast(device_type=device):
            outputs = model(
                image_tensor,
                prompt_ids=pad_prompt,
                prompt_mask=pad_mask,
                cap_ids=cap_ids,
                cap_mask=cap_mask
            )

    generated_ids = outputs.logits.argmax(dim=-1)
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 清理显存
    del outputs, image_tensor, cap_ids, cap_mask, prompt
    torch.cuda.empty_cache()
    gc.collect()

    return caption


def main():
    device = args.device
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, device)
    image_tensor = preprocess_image(args.image_path, device)
    caption = generate_caption(model, tokenizer, image_tensor, device)
    print("\n>> Generated Caption:")
    print(caption)


if __name__ == "__main__":
    main()
