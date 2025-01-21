from transformers import PreTrainedTokenizerFast
import os
from collections import namedtuple
import pickle
from tokenizers import models, pre_tokenizers, trainers, Tokenizer, processors, decoders, normalizers

language_set = namedtuple("language", ["source", "target"])
spec_tokens = namedtuple("special_tokens", ["unk", "sos", "eos", "pad"])

class translation_tokenizer:

    def __init__(self, vocab_size, max_sequence_len):
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.special_tokens = spec_tokens(unk="[UNK]", sos="[SOS]", eos="[EOS]", pad="[PAD]")
        self.src_save_name = "tokenizer_source.json"
        self.trg_save_name = "tokenizer_target.json"
    
    def get_tokenizer(self, path):
        path_to_file = os.path.abspath(os.path.join(path, self.src_save_name))
        self.src_wrap = PreTrainedTokenizerFast.from_pretrained(path_to_file)
        path_to_file = os.path.abspath(os.path.join(path, self.trg_save_name ))
        self.trg_wrap = PreTrainedTokenizerFast.from_pretrained(path_to_file)


        with open(os.path.join(path, "tok_vars.pickle"), 'rb') as f:
            _vars = pickle.load(f)
        self.sos_token_ids, self.eos_token_ids, self.pad_token_ids = _vars["sos"], _vars["eos"], _vars["pad"]

    def save_tokenizer(self, path):
        path_to_file = os.path.abspath(os.path.join(path, self.src_save_name ))
        self.src_wrap.save_pretrained(path_to_file)
        path_to_file = os.path.abspath(os.path.join(path, self.trg_save_name ))
        self.trg_wrap.save_pretrained(path_to_file)

        _vars = {"sos":self.sos_token_ids, "eos": self.eos_token_ids, "pad":self.pad_token_ids}
        with open(os.path.join(path, "tok_vars.pickle"), 'wb') as handle:
            pickle.dump(_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self, src_sentence, trg_sentence):
        tokens_src = self.src_wrap(src_sentence, padding='max_length', max_length=self.max_sequence_len , truncation=True)
        tokens_trg = self.trg_wrap(trg_sentence, padding='max_length', max_length=self.max_sequence_len , truncation=True)
        return tokens_src["input_ids"], tokens_trg["input_ids"]
    
    def __len__(self):
        return len(self.wrap.vocab)

    def decode(self, sentence, key="target"):
        if key == "target":
            return self.trg_wrap.decode(sentence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        else:
            return self.src_wrap.decode(sentence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    def pad_id(self):
        return self.pad_token_ids 
    
    def sos_id(self):
        return self.sos_token_ids 
    
    def eos_id(self):
        return self.eos_token_ids 

    def set_tokenizers(self, train_data:language_set):
        self.src_wrap, self.sos_token_ids, self.eos_token_ids, self.pad_token_ids = self.set_tokenizer(train_data.source)
        self.trg_wrap, _, _, _ = self.set_tokenizer(train_data.target)

    def set_tokenizer(self, train_data:language_set):
        
        tokenizer = Tokenizer(models.WordPiece(unk_token=self.special_tokens.unk))
        
        # normalizers
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
        # split on white space and punctuation
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence( [pre_tokenizers.WhitespaceSplit(),
                                                            pre_tokenizers.Punctuation()])
                
        # build tokenization pipeline
        trainer = trainers.WordPieceTrainer(vocab_size=self.vocab_size, 
                                            special_tokens=[*self.special_tokens],
                                            min_frequency=1)


        tokenizer.train_from_iterator(train_data, trainer=trainer)

        sos_token_id =  tokenizer.token_to_id(self.special_tokens.sos) # token at the beginning
        eos_token_id =  tokenizer.token_to_id(self.special_tokens.eos) # token at the end
        pad_tokedn_id = tokenizer.token_to_id(self.special_tokens.pad) # token for padding


        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.special_tokens.sos}:0 $A:0 {self.special_tokens.eos}:0",
            pair = f"{self.special_tokens.sos}:0 $A:0 {self.special_tokens.eos}:0 $B:1 {self.special_tokens.eos}:1",
            special_tokens=[(self.special_tokens.sos, sos_token_id), (self.special_tokens.eos, eos_token_id)],
        )

        tokenizer.decoder = decoders.WordPiece()
        
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token=self.special_tokens.unk,
            pad_token=self.special_tokens.pad,
            cls_token=self.special_tokens.sos,
            sep_token=self.special_tokens.eos,
            mask_token="[MASK]",
            padding=True
        )

        return wrapped_tokenizer, sos_token_id, eos_token_id, pad_tokedn_id