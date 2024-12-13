from rouge_score import rouge_scorer

def calculate_rouge_score(input_text, decoded_output_without_watermark, decoded_output_with_watermark, human_reference_text):
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # ROUGE Scores for multiple references
    rouge_references = []
    for reference in human_reference_text:
        rouge_references.append(scorer.score(reference, input_text))

    rouge_without_watermark = scorer.score(input_text, decoded_output_without_watermark)
    rouge_with_watermark = scorer.score(input_text, decoded_output_with_watermark)

    # Average ROUGE scores across references
    avg_rouge_references = {
        metric: sum([score[metric].fmeasure for score in rouge_references]) / len(rouge_references)
        for metric in rouge_references[0]
    }

    return avg_rouge_references, rouge_without_watermark, rouge_with_watermark
