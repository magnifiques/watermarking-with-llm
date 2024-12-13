import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction 
 
def calculate_bleu_score(input_text, decoded_output_without_watermark, decoded_output_with_watermark, human_reference_text): 
    # Tokenize the reference texts 
    reference_tokens = [nltk.word_tokenize(input_text.lower().strip())]
 
    # Tokenize the candidate outputs (without and with watermark) 
    candidate_without_watermark = nltk.word_tokenize(decoded_output_without_watermark.lower().strip()) 
    candidate_with_watermark = nltk.word_tokenize(decoded_output_with_watermark.lower().strip()) 
 
    # Prepare the human references: list of tokenized human references for BLEU calculation 
    human_reference_tokens = [nltk.word_tokenize(ref.lower().strip()) for ref in human_reference_text] 
 
    # Apply smoothing to avoid zero scores 
    chencherry = SmoothingFunction() 
 
    # Calculate BLEU Scores for each case (compare candidate vs references) 
    bleu_without_watermark = sentence_bleu(reference_tokens, candidate_without_watermark, smoothing_function=chencherry.method1) 
    bleu_with_watermark = sentence_bleu(reference_tokens, candidate_with_watermark, smoothing_function=chencherry.method1) 
 
    # Calculate BLEU Score for reference vs itself 
    bleu_with_references = sum(sentence_bleu(reference_tokens[0], ref, smoothing_function=chencherry.method1) for ref in human_reference_tokens) / len(human_reference_tokens)

    return bleu_with_references, bleu_without_watermark, bleu_with_watermark