from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Ensure NLTK's punkt tokenizer is downloaded
nltk.download('punkt_tab')

# input_text = "Donkey Kong Country is a side-scrolling platform game.[2] A reboot of the Donkey Kong franchise,[3][4] its story begins when King K. Rool and his army of crocodiles, the Kremlings, steal the Kongs' banana hoard.[5][6] The gorilla Donkey Kong and his nephew Diddy Kong set out to reclaim the hoard and defeat the Kremlings.[6] Donkey and Diddy serve as the player characters of the single-player game; they run alongside each other and the player can swap between them at will. Donkey is stronger and can defeat enemies more easily; Diddy is faster and more agile.[7] Both can walk, run, jump, pick up and throw objects, and roll; Donkey can slap the terrain to defeat enemies or find items.[8] The player begins in a world map that tracks their progress and provides access to the 40 levels.[9][10] The player attempts to complete each level while traversing the environment, jumping between platforms, and avoiding enemy and inanimate obstacles. Level themes include jungles, underwater reefs, caves, mines, mountains, and factories.[11] Some feature unique game mechanics, such as rideable minecarts, blasting out of cannons resembling barrels, and swinging ropes.[12] Each area ends with a boss fight with a large enemy.[13] Donkey and Diddy can"

def calculate_bleu_score(input_text, decoded_output_without_watermark, decoded_output_with_watermark):
    

    # Reference text (input_text or a modified version of it)
    reference = nltk.word_tokenize(input_text.lower())

    # Candidate outputs
    candidate_without_watermark = nltk.word_tokenize(decoded_output_without_watermark.lower())
    candidate_with_watermark = nltk.word_tokenize(decoded_output_with_watermark.lower())

    # Use smoothing to avoid zero scores
    chencherry = SmoothingFunction()
    
    # BLEU Scores
    bleu_without_watermark = sentence_bleu(
        [reference], candidate_without_watermark, smoothing_function=chencherry.method1
    )
    bleu_with_watermark = sentence_bleu(
        [reference], candidate_with_watermark, smoothing_function=chencherry.method1
    )
    return bleu_without_watermark, bleu_with_watermark
    # print(f"BLEU Score (Without Watermark): {bleu_without_watermark}")
    # print(f"BLEU Score (With Watermark): {bleu_with_watermark}")