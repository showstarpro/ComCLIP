# Modified from:
# https://github.com/tsb0601/MMVP/blob/main/scripts/evaluate_vlm.py
# https://github.com/baaivision/DIVA/blob/main/run_DIVA_with_*.py

import os
import argparse
import csv
import torch
import clip
import open_clip
from tqdm import tqdm
from PIL import Image


def main(model_name, pretrained, benchmark_dir, load_full_model=False):
    # Load original model
    if model_name == "SigLip":
        model, preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:timm/ViT-SO400M-14-SigLIP', device='cpu'
        )
        tokenize_func = open_clip.get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP')
    else:
        model, preprocess = open_clip.create_model_from_pretrained(
                model_name, pretrained='openai', device='cpu'
            )
        tokenize_func = open_clip.tokenize
    clip_fwd_func = openclip_fwd

    if torch.cuda.is_available():
        model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Eval
    print("Original performance:")
    results_original = benchmark_model(preprocess, model, tokenize_func, clip_fwd_func, benchmark_dir)
    print(results_original)

    # Load finetuned model
    state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
    if load_full_model:
        print("### Load the full model including text encoder and vision encoder")
        if 'vision_encoder_state_dict' in state_dict.keys():
            model.visual.load_state_dict(state_dict['vision_encoder_state_dict'])
            if 'text_encoder_state_dict' in state_dict:
                model.text.load_state_dict(state_dict['text_encoder_state_dict'])
        else:
            model.load_state_dict(state_dict, strict=False)
    else:
        print("### Only load the vision encoder")
        if 'vision_encoder_state_dict' in state_dict.keys():
            # tecoa checkpoint
            model.visual.load_state_dict(state_dict['vision_encoder_state_dict'])
        else:
            has_visual_prefix = any(k.startswith('visual.') for k in state_dict.keys())
            if has_visual_prefix:
                # 完整 CLIP checkpoint → 提取 visual 部分并去掉 visual. 前缀
                visual_state_dict = {
                    k.replace('visual.', ''): v 
                    for k, v in state_dict.items() 
                    if k.startswith('visual.')
                }
                model.visual.load_state_dict(visual_state_dict)
                print(f"  Extracted {len(visual_state_dict)} visual keys from full CLIP checkpoint")
            else:
                model.visual.load_state_dict(state_dict)

    # Eval
    print("\nAfter Post-training:")
    results_after = benchmark_model(preprocess, model, tokenize_func, clip_fwd_func, benchmark_dir)
    print(results_after)


@torch.no_grad()
def clip_fwd(model, imgs, text1, text2):
    logits_per_image1, logits_per_text1 = model(imgs, text1)
    logits_per_image2, logits_per_text2 = model(imgs, text2)
    return logits_per_text1, logits_per_text2


@torch.no_grad()
def openclip_fwd(model, imgs, text1, text2):
    returned_tuple = model(imgs, text1) # len(returned_tuple) is 3 (4) for openclip (siglip)
    image_features = returned_tuple[0]
    text1_features = returned_tuple[1]
    returned_tuple = model(imgs, text2)
    image_features = returned_tuple[0]
    text2_features = returned_tuple[1]
    logits_per_image1 = 100.0 * image_features @ text1_features.T
    logits_per_text1 = logits_per_image1.T
    logits_per_image2 = 100.0 * image_features @ text2_features.T
    logits_per_text2 = logits_per_image2.T
    return logits_per_text1, logits_per_text2


@torch.no_grad()
def benchmark_model(preprocess, model, tokenize_func, clip_fwd_func, benchmark_dir, csv_outfile=None):
    device = next(model.parameters()).device

    image_dir = os.path.join(benchmark_dir, 'MLLM_VLM Images')
    csv_file = os.path.join(benchmark_dir, 'Questions.csv')

    if csv_outfile is not None:
        csv_writer = csv.writer(csv_outfile)
        csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score']) # header

    categories = [
        'Orientation and Direction', 'Presence of Specific Features',
        'State and Condition', 'Quantity and Count',
        'Positional and Relational Context', 'Color and Appearance',
        'Structural Characteristics', 'Texts',
        'Viewpoint and Perspective'
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in tqdm(enumerate(reader)):
            qid1, qtype1, statement1 = row

            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row

            qid1, qid2 = int(qid1), int(qid2)

            img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg'))
            img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg'))

            text1 = 'a photo of ' + statement1
            text2 = 'a photo of ' + statement2

            text1 = tokenize_func([text1]).to(device)
            text2 = tokenize_func([text2]).to(device)

            img1 = preprocess(img1).unsqueeze(0).to(device)
            img2 = preprocess(img2).unsqueeze(0).to(device)
            imgs = torch.cat((img1, img2), dim=0)

            logits_per_text1, logits_per_text2 = clip_fwd_func(model, imgs, text1, text2)
            probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
            probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

            img1_score1 = probs1[0][0]
            img1_score2 = probs2[0][0]

            pred1 = "img1" if img1_score1 > 0.5 else "img2"
            pred2 = "img1" if img1_score2 > 0.5 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            if csv_outfile is not None:
                csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])

            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1

        if csv_outfile is not None:
            csv_outfile.close()

    # Calculate percentage accuracies
    category_score_list = []
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100
        category_score_list.append(pair_accuracies[category])
    pair_accuracies['average_score'] = sum(category_score_list) / len(category_score_list)
    return pair_accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ViT-L-14",
                        help="Model name for posttrained model checkpoint")
    parser.add_argument("--pretrained", type=str, default="/path/to/checkpoints/coco-cliprefine-vitl-seed80-bs64x8.pt",
                        help="Path to un2CLIP finetuned model checkpoint")
    parser.add_argument("--benchmark_dir", type=str, default="/path/to/data/MMVP__MMVP_VLM/main",
                        help="Path to MMVP_VLM benchmark dataset directory")
    parser.add_argument('--load_full_model', action='store_true', help="load both vision and text encoder weights from checkpoint")
    args = parser.parse_args()
    print(args)
    main(args.model_name, args.pretrained, args.benchmark_dir, args.load_full_model)
