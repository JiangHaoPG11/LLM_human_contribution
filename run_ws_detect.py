from dataloader import load_data
from watermarker.LLM_watermarker import GPTWatermarkDetector
import argparse
from tqdm import tqdm
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LogitsProcessorList



parser = argparse.ArgumentParser()

# parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
# parser.add_argument("--model_name", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
parser.add_argument("--tokenizer_name", type=str, default='bert-base-uncased')

parser.add_argument("--fraction", type=float, default=0.5)
parser.add_argument("--strength", type=float, default=2.0)
parser.add_argument("--wm_key", type=int, default=0)
parser.add_argument("--max_new_tokens", type=int, default=128)

parser.add_argument("--num_test", type=int, default=5)
parser.add_argument("--beam_size", type=int, default=None)
parser.add_argument("--top_k", type=int, default=None)
parser.add_argument("--top_p", type=float, default=0.9)

parser.add_argument("--test_min_tokens", type=int, default=200)

parser.add_argument("--polishedText_pth", type=str, default="./data_generation/PolishedText.csv")
parser.add_argument("--generatedText_pth", type=str, default="./data_generation/GeneratedText.csv")

args = parser.parse_args()

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)
    vocab_size = tokenizer.vocab_size

    detector = GPTWatermarkDetector(fraction=args.fraction,
                                    strength=args.strength,
                                    vocab_size=vocab_size,
                                    watermark_key=args.wm_key)
    z_score_list = []

    # 加载数据
    datas = load_data(args)
    generateText = datas['text']
    generateLabel = datas['label']

    outputs = []

    for idx in tqdm(range(len(generateText)), total=min(len(generateText), args.test_min_tokens)):
        text = generateText[idx]
        label = generateLabel[idx]
        # 分词
        gen_tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

        z_score_list.append(detector.detect(gen_tokens))

    print('Finished!')




if __name__ == "__main__":

    main(args)