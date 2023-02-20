from transformers import CLIPTokenizer

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", 
                                              TOKENIZERS_PARALLELISM=False)
frequent_words = "in on of to this that which video image picture I we can see a an the \
            about for and from into here there is are . , <|endoftext|> <|startoftext|>"
frequent_ids = clip_tokenizer.convert_tokens_to_ids(clip_tokenizer.tokenize(frequent_words))
