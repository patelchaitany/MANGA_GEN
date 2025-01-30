from typing import List
import Levenshtein

def calculate_levenshtein_similarity(str1: str, str2: str) -> float:
    """Calculate similarity percentage using Levenshtein distance"""
    distance = Levenshtein.distance(str1.lower(), str2.lower())
    max_len = max(len(str1), len(str2))
    similarity = (1 - distance / (max_len+1)) * 100
    return similarity

def word_by_word_levenshtein(text1: str, text2: str) -> dict:
    """Calculate Levenshtein distance word by word"""
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    
    distances = {}
    for i, word1 in enumerate(words1):
        if i < len(words2):
            distance = Levenshtein.distance(word1, words2[i])
            distances[f"Word {i+1}"] = {
                "word1": word1,
                "word2": words2[i],
                "distance": distance
            }
    
    return distances 