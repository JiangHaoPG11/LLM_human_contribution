from dataloader import load_data
from watermarker.LLM_watermarker import GPTWatermarkLogitsWarper
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


def main(args):
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()

    watermark_processor = LogitsProcessorList([GPTWatermarkLogitsWarper(fraction=args.fraction,
                                                                        strength=args.strength,
                                                                        vocab_size=model.config.vocab_size,
                                                                        watermark_key=args.wm_key)])
    
    # 加载数据
    datas = load_data(args)
    totalText = datas['text']
    totalLabel = datas['label']

    outputs = []

    for idx in tqdm(range(len(totalText)), total=min(len(totalText), args.num_test)):
        text = totalText[idx]
        label = totalLabel[idx]
        # 分词
        batch = tokenizer(text, truncation=True, return_tensors="pt")
        num_tokens = len(batch['input_ids'][0])

        with torch.no_grad():
            generate_args = {
                **batch,
                'logits_processor': watermark_processor,
                'output_scores': True,
                'return_dict_in_generate': True,
                'max_new_tokens': args.max_new_tokens,
            }

            if args.beam_size is not None:
                generate_args['num_beams'] = args.beam_size
            else:
                generate_args['do_sample'] = True
                generate_args['top_k'] = args.top_k
                generate_args['top_p'] = args.top_p

            generation = model.generate(**generate_args)
            gen_text = tokenizer.batch_decode(generation['sequences'][:, num_tokens:], skip_special_tokens=True)
        print(gen_text)
        
    print("Finished!")


if __name__ == "__main__":

    main(args)