from rouge_score import rouge_scorer

# input_text = "Donkey Kong Country is a side-scrolling platform game.[2] A reboot of the Donkey Kong franchise,[3][4] its story begins when King K. Rool and his army of crocodiles, the Kremlings, steal the Kongs' banana hoard.[5][6] The gorilla Donkey Kong and his nephew Diddy Kong set out to reclaim the hoard and defeat the Kremlings.[6] Donkey and Diddy serve as the player characters of the single-player game; they run alongside each other and the player can swap between them at will. Donkey is stronger and can defeat enemies more easily; Diddy is faster and more agile.[7] Both can walk, run, jump, pick up and throw objects, and roll; Donkey can slap the terrain to defeat enemies or find items.[8] The player begins in a world map that tracks their progress and provides access to the 40 levels.[9][10] The player attempts to complete each level while traversing the environment, jumping between platforms, and avoiding enemy and inanimate obstacles. Level themes include jungles, underwater reefs, caves, mines, mountains, and factories.[11] Some feature unique game mechanics, such as rideable minecarts, blasting out of cannons resembling barrels, and swinging ropes.[12] Each area ends with a boss fight with a large enemy.[13] Donkey and Diddy can"


def calculate_rouge_scorer(input_text, decoded_output_without_watermark, decoded_output_with_watermark): 
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # ROUGE Scores
    rouge_without_watermark = scorer.score(input_text, decoded_output_without_watermark)
    
    rouge_with_watermark = scorer.score(input_text, decoded_output_with_watermark)
    return rouge_without_watermark, rouge_with_watermark
    # print("ROUGE Score (Without Watermark):", rouge_without_watermark)
    # print("ROUGE Score (With Watermark):", rouge_with_watermark)
