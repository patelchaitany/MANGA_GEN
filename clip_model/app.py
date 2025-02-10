from flask import Flask, request, jsonify
from PIL import Image
import requests
from multiprocessing import Pool
from functools import partial
import base64
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

model = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M")
processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M")


@app.route('/relevance', methods=['POST'])
def get_relevance():
    data = request.get_json()
    #print(f"Received JSON data: {data}")  # Log received JSON data
    if not data or 'prompts' not in data or 'images' not in data:
        return jsonify({'error': 'Prompts and images are required in JSON body'}), 400

    prompts = data['prompts']
    image_base64_strings = data['images']
    print(f"Received prompts: {prompts}")  # Log received prompts
    print(f"Received image_base64_strings (count): {len(image_base64_strings)}")  # Log received image_base64_strings

    if not isinstance(prompts, list):
        return jsonify({'error': 'Prompts must be a list'}), 400
    if not isinstance(image_base64_strings, list):
        return jsonify({'error': 'Images must be a list of base64 strings'}), 400

    images = []
    for image_base64_string in image_base64_strings:
        try:
            image_data = base64.b64decode(image_base64_string)
            image_stream = BytesIO(image_data)
            image = Image.open(image_stream)
            images.append(image)
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {e}'}), 400

    results = []
    
    image_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        clip_text_embeddings = model.get_text_features(image_inputs["input_ids"])
        clip_text_embeddings = clip_text_embeddings / clip_text_embeddings.norm(
            dim=-1, keepdim=True
        )
        clip_image_embeddings = model.get_image_features(image_inputs["pixel_values"])
        clip_image_embeddings = clip_image_embeddings / clip_image_embeddings.norm(
            dim=-1, keepdim=True
        )
        clip_cos_sim = torch.mm(clip_text_embeddings, clip_image_embeddings.T)
    print(clip_cos_sim)
    # print(f"output {outputs}")
    # logits_per_image = outputs.logits_per_image # Logits for each prompt for each image
    # print(f"Logits per image: {logits_per_image}")
    # probs = logits_per_image.softmax(dim=1).tolist() # Softmax across prompts for each image
    print(f"Relevance scores: {clip_cos_sim}")
    
    highest_similarity_results = []
    for i in range(clip_cos_sim.size(0)):  # Iterate over prompts
        prompt_similarities = clip_cos_sim[i]
        max_similarity_index = torch.argmax(prompt_similarities)
        max_similarity_score = prompt_similarities[max_similarity_index].item()
        highest_similarity_results.append({
            'prompt_index': i,
            'highest_similarity_score': max_similarity_score,
            'highest_similarity_image_index': max_similarity_index.item()
        })

    return jsonify({'highest_similarity_results': highest_similarity_results})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

# removed process_image_prompt_pair function
# def process_image_prompt_pair(prompt, image, processor, model):
#     print(f"Processing prompt: {prompt}, image: {image.format} {image.size}") # Log input to process_image_prompt_pair
#     inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
#     outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image
#     print(f"Logits per image: {logits_per_image}") # Log logits_per_image
#     probs = logits_per_image.softmax(dim=1).tolist()
#     print(f"Relevance scores: {probs}") # Log output of process_image_prompt_pair
#     return probs[0]

