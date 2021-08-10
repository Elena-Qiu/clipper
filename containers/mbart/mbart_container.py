from __future__ import print_function
import os
import sys
import pickle
import argparse

import rpc

from transformers import MBartForConditionalGeneration, MBartTokenizer


class MbartContainer(rpc.ModelContainerBase):
    def __init__(self, src_lang, dst_lang):
        self.src_lang = src_lang
        self.dst_lang = dst_lang

        print("Loading model and tokenizer", flush=True)
        self.mbart_model= MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
        self.tokenizer = MBartTokenizer.from_pretrained(
            "facebook/mbart-large-en-ro",
            src_lang=src_lang, tgt_lang=dst_lang
        )
        print("Done loading model and tokenizer.", flush=True)

    def predict_strings(self, inputs):
        results = []
        for string in inputs:
            encoded_hi = self.tokenizer(string, return_tensors="pt")
            generated_tokens = self.mbart_model.generate(
                **encoded_hi,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.dst_lang]
            )
            result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            results.append(result)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rpc', action='store_true', default=False)
    parser.add_argument('--src', default='en_XX')
    parser.add_argument('--dst', default='zh_CN')
    args = parser.parse_args()

    model = MbartContainer(args.src, args.dst)
    if args.rpc:
        rpc_service = rpc.RPCService()
        rpc_service.start(model)
