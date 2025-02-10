import cv2
import numpy as np
import itertools
import requests
from io import BytesIO
import asyncio
import logging
from termcolor import colored
from PIL import Image, UnidentifiedImageError
from src.search_utils.search import perform_search, crawl_website
lock = asyncio.Lock()  # Create an async lock

import base64
import json
import os
def generate_permutations(hash_array, k=4):
    """Generate block permutations from hash array"""
    block_size = len(hash_array) // k
    blocks = [hash_array[i*block_size:(i+1)*block_size] for i in range(k)]
    
    permutations = []
    for perm in itertools.permutations(blocks):
        permutations.append(np.concatenate(perm))
    return permutations 

def pdq_hash(image_path, k=4):
    # Step 0: Read image from BytesIO
    if isinstance(image_path, bytes):
        image_array = np.frombuffer(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        # Fallback to file path reading
        image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Could not decode image from bytes or file path")
        
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (64, 64))

    # Step 1: Compute the Discrete Cosine Transform (DCT)
    dct = cv2.dct(np.float32(resized_image))

    # Step 2: Select a Low-Frequency Subset
    low_freq_subset = dct[:8, :8]  # Taking the top-left 8x8 block

    # Step 3: Compute a Threshold Using the Median
    median_value = np.median(low_freq_subset)

    # Step 4: Generate the Binary Hash
    binary_hash = (low_freq_subset > median_value).astype(int).flatten()
    
    # Generate permutations for efficient search
    permutations = generate_permutations(binary_hash, k)
    
    return binary_hash, permutations

def hamming_distance(hash1, hash2):
    return np.sum(hash1 != hash2)


class DocumentIndex:
    """Index for storing and querying document hashes"""
    def __init__(self):
        self.index = {}
        self.documents = {}
        self._lock = asyncio.Lock()  # Initialize asyncio lock

    async def add_document(self, doc_id, hash_permutations, uniqueness_threshold=10):
        """Add document only if it's unique based on Hamming distance, using async lock"""
        new_hash = hash_permutations[0]

        async with self._lock: # Acquire async lock
            # Check against existing documents
            for existing_id, existing_hash in self.documents.items():
                if hamming_distance(existing_hash, new_hash) <= uniqueness_threshold:
                    return False  # Document is not unique

            # Add if unique
            self.documents[doc_id] = new_hash
            print(f"{doc_id} is uniqe")
            for perm in hash_permutations:
                self.index.setdefault(tuple(perm), []).append(doc_id)
            return True
            
    def query(self, query_permutations, threshold=5):
        """Find similar documents using permuted hashes"""
        candidates = {}
        for perm in query_permutations:
            for doc_id in self.index.get(tuple(perm), []):
                candidates[doc_id] = candidates.get(doc_id, 0) + 1
                
        # Get documents with most permutation matches
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        
        # Verify Hamming distance for top candidates
        results = []
        for doc_id, count in sorted_candidates:
            distance = hamming_distance(self.documents[doc_id], query_permutations[0])
            if distance <= threshold:
                results.append((doc_id, distance))
                
        return sorted(results, key=lambda x: x[1])


async def fetch_image(url):
    """Fetch image bytes from URL asynchronously."""
    try:
        response = requests.get(url, timeout=1)
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            logging.warning(f"Failed to fetch image from {url}: Status code {response.status_code}")
            return None
    except requests.RequestException as e:
        logging.warning(f"Error fetching image from {url}: {e}")
        return None



async def perform_search_and_filter(query):
    valid_image_urls = []  # List to store valid image URLs
    unique_hashes = set()  # Set to track unique image hashes
    """Perform search and filter valid image URLs and fetch image bytes."""
    results = await perform_search(query)

    index = DocumentIndex()  # Create an instance of DocumentIndex
    clip_threshold = 0 # Threshold for CLIP relevance

    async def process_item(item):
        url = item['link']
        try:
            print(colored(f"url : {url}","green"))
            response = await asyncio.to_thread(requests.get, url, allow_redirects=True, timeout=1)
            if response.status_code == 200:
                img_bytes = BytesIO(response.content)
                try:
                    img = Image.open(img_bytes)  # Verify it's a valid image
                    image_hash, perms = pdq_hash(img_bytes.getvalue())  # Get hash and permutations
                    if await index.add_document(url, perms):  # Check for uniqueness
                        # Encode image to base64 for CLIP API
                        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                        clip_data = {'prompts': [query], 'images': [img_base64]}
                        clip_api_url = os.environ.get('CLIP_API_URL', "http://0.0.0.0:5000/relevance") # Use CLIP_API_URL from .env or default
                        try:
                            clip_response = requests.post(clip_api_url, json=clip_data, timeout=2)
                            if clip_response.status_code == 200:
                                relevance_data = clip_response.json()
                                highest_similarity_score = relevance_data['highest_similarity_results'][0]['highest_similarity_score']
                                if highest_similarity_score >= clip_threshold:
                                    valid_image_urls.append(url)  # Store URL only if relevant
                                    logging.info(f"Valid, unique and relevant image URL found: {url} (Relevance Score: {highest_similarity_score})")
                                else:
                                    logging.info(f"Image URL found but not relevant, skipping: {url} (Relevance Score: {highest_similarity_score})")
                            else:
                                logging.warning(f"CLIP API request failed with status code: {clip_response.status_code}")
                        except requests.RequestException as e:
                            logging.warning(f"Error calling CLIP API: {e}")
                    else:
                        logging.info(f"Duplicate image found, skipping: {url}")
                except UnidentifiedImageError:
                    print(f"Invalid image format for URL: {url}")
            else:
                print(f"Invalid image URL: {url}")
        except requests.RequestException:
            print(f"Invalid image URL: {url}")

    tasks = [process_item(item) for item in results]
    await asyncio.gather(*tasks)
    logging.info("Valid image URLs saved.") # Removed Redis specific log
    return valid_image_urls  # Return URLs


def check_url_validity(url):

    try:
        response = requests.head(url, allow_redirects=True, timeout=1)
        #print(colored(f"url {url} {response}","green"))
        return response.status_code == 200
    except requests.RequestException:
        return False


# Example usage with improved comparison
if __name__ == "__main__":
    # Create index and add documents
    index = DocumentIndex()
    
    # Example with URL fetching and BytesIO
    url1 = 'https://static.animecorner.me/2025/01/1735819266-9927964e231a6bc64c4e1b4c4151b820-150x150.png'
    url2 = 'https://static.animecorner.me/2025/01/1735819266-9927964e231a6bc64c4e1b4c4151b820-300x169.png'
    # Fetch and process first image
    response = requests.get(url1)
    img1_bytes = BytesIO(response.content).getvalue()
    orig_hash1, perms1 = pdq_hash(img1_bytes)
    index.add_document("doc1", perms1)
    
    # Fetch and process second image
    response = requests.get(url2)
    img2_bytes = BytesIO(response.content).getvalue()
    orig_hash2, perms2 = pdq_hash(img2_bytes)
    index.add_document("doc2", perms2)
    
    # Query the index
    results = index.query(perms1, threshold=5)
    
    print("Similar documents:")
    for i in index.documents:
        print(f"Key {i}")
    for doc_id, distance in results:
        print(f"Document {doc_id} - Hamming distance: {distance}")
