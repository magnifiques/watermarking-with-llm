import torch
from transformers import LogitsProcessor, LogitsProcessorList
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
# from demo_watermark import create_cluster

class ClusterBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, token_to_cluster, bias_strength=1.5):
        self.token_to_cluster = token_to_cluster
        self.bias_strength = bias_strength
        self.prev_token_cluster = None

    def __call__(self, input_ids, scores):
        # Retrieve cluster of the last token generated
        last_token_id = input_ids[0, -1].item()
        self.prev_token_cluster = self.token_to_cluster.get(last_token_id, None)
        
        if self.prev_token_cluster is not None:
            # Apply bias to tokens in the same cluster
            bias_vector = torch.ones_like(scores).to(scores.device)

            for token_id, cluster_id in self.token_to_cluster.items():
                # Ensure token_id is within bounds of the vocabulary size
                if token_id < scores.size(1) and cluster_id == self.prev_token_cluster:
                    bias_vector[0, token_id] *= self.bias_strength

            scores = scores * bias_vector

        return scores

    
# def generate_without_watermark_cluster(prompt, args, model=None, device=None, tokenizer=None):
#     """Generate text without watermark, for comparison."""
#     gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

#     if args.use_sampling:
#         gen_kwargs.update(dict(
#             do_sample=True, 
#             top_k=0,
#             temperature=args.sampling_temp
#         ))
#     else:
#         gen_kwargs.update(dict(
#             num_beams=args.n_beams
#         ))

#     generate_without_watermark = partial(
#         model.generate,
#         **gen_kwargs
#     )
    
#     # Tokenize the input
#     tokd_input = tokenizer(prompt, return_tensors="pt").to(device)
    
#     # Generate without watermark
#     output_without_watermark = generate_without_watermark(**tokd_input)

#     # Decode the output
#     decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    
#     return decoded_output_without_watermark


# def generate_with_watermark_cluster(prompt, args, model=None, device=None, tokenizer=None):
#     """Generate text with watermark and cluster bias."""
#     # Generate the token-to-cluster mapping
#     token_to_cluster = create_cluster(model, tokenizer, args)
    
#     # Initialize processors
#     watermark_processor = WatermarkLogitsProcessor(
#         vocab=list(tokenizer.get_vocab().values()),
#         gamma=args.gamma,
#         delta=args.delta,
#         seeding_scheme=args.seeding_scheme,
#         select_green_tokens=args.select_green_tokens
#     )
    
#     cluster_bias_processor = ClusterBiasLogitsProcessor(
#         token_to_cluster=token_to_cluster,
#         bias_strength=1.5
#     )

#     # Combine processors
#     generate_with_watermark = partial(
#         model.generate,
#         logits_processor=LogitsProcessorList([watermark_processor, cluster_bias_processor]), 
#         max_new_tokens=args.max_new_tokens,
#         do_sample=True,
#         temperature=args.sampling_temp
#     )

#     # Tokenize the input
#     tokd_input = tokenizer(prompt, return_tensors="pt").to(device)
    
#     # Generate with watermark
#     output_with_watermark = generate_with_watermark(**tokd_input)
    
#     # Decode the output
#     decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
    
#     return decoded_output_with_watermark


# def generate_with_cluster(prompt, args, model=None, device=None, tokenizer=None):
#     """Main generate function to handle both watermarked and non-watermarked generations."""
    
#     # Generate output without watermark
#     decoded_output_without_watermark = generate_without_watermark_cluster(prompt, args, model, device, tokenizer)
    
#     # Generate output with watermark
#     decoded_output_with_watermark = generate_with_watermark_cluster(prompt, args, model, device, tokenizer)
    
#     return decoded_output_without_watermark, decoded_output_with_watermark


