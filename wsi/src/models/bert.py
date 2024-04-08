from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np


def get_batch(collection, chunk_size=5):
    for i in range(0, len(collection), chunk_size):
        yield collection[i:i + chunk_size]


class Bert:
    def __init__(self, config):
        self.device = torch.device(f'cuda:{config.cuda_device}') if config.cuda_device >= 0 else torch.device('cpu')
        self.config = config

        with torch.no_grad():
            model = AutoModelForMaskedLM.from_pretrained(config.model, output_hidden_states=True)
            model.to(self.device)
            model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(config.model)
            self.model = model
            self.lemmatized_vocab = []
            self.original_vocab = []
            vocab = sorted(self.tokenizer.vocab.items(), key=lambda x: x[1])

            import pymorphy3
            nlp = pymorphy3.MorphAnalyzer()
            for token, _ in vocab:
                lemma = nlp.parse(token)[0].normal_form
                self.lemmatized_vocab.append(lemma)
                self.original_vocab.append(token.lower())

    def get_formated_sent(self, pre, target, post, pattern):
        replacements = dict(pre=pre, target=target, post=post)
        for predicted_token in ['{mask_predict}', '{target_predict}']:
            if predicted_token in pattern: 
                before_pred, after_pred = pattern.split(predicted_token)
                before_pred = [self.tokenizer.cls_token] + self.tokenizer.tokenize(before_pred.format(**replacements))
                after_pred = self.tokenizer.tokenize(after_pred.format(**replacements)) + [self.tokenizer.sep_token]
                target_prediction_idx = len(before_pred)
                target_tokens = [self.tokenizer.mask_token] if predicted_token == '{mask_predict}' else self.tokenizer.tokenize(target)
                return before_pred + target_tokens + after_pred, target_prediction_idx

    def predict_substitutes(self, inst_id_to_sentence):
        pattern = self.config.pattern
        res = {}
        with torch.no_grad():
            sorted_by_len = sorted(inst_id_to_sentence.items(), key=lambda x: len(x[1][0]) + len(x[1][2]))
            for batch in get_batch(sorted_by_len, self.config.max_batch_size):
                batch_sents, target_ids = [], []
                for _, (pre, target, post) in batch:
                    tokens, target_id = self.get_formated_sent(pre, target, post, pattern)
                    batch_sents.append(tokens)
                    target_ids.append(target_id)

                tokens_idx = [self.tokenizer.convert_tokens_to_ids(item) for item in batch_sents]
                max_len = len(max(tokens_idx, key=lambda x: len(x)))

                input_batch = torch.zeros((len(tokens_idx), max_len), dtype=torch.long)
                for idx, row in enumerate(tokens_idx):
                    input_batch[idx, 0:len(row)] = torch.tensor(row)
                input_batch = input_batch.clone().detach().to(self.device)
                attention_mask = input_batch != 0

                logits = self.model(input_batch, attention_mask=attention_mask).logits
                targets_logits = logits[range(logits.shape[0]), target_ids, :]

                topk_vals, topk_idxs = torch.topk(targets_logits, self.config.prediction_cutoff, -1)

                probs_batch = torch.softmax(topk_vals, -1).detach().cpu().numpy()

                topk_idxs_batch = topk_idxs.detach().cpu().numpy()

                for (inst_id, (pre, target, post)), probs, topk_idxs in zip(batch, probs_batch, topk_idxs_batch):
                    lemma = target.lower()
                    probs = probs.copy()
                    target_vocab = self.lemmatized_vocab

                    for i in range(self.config.prediction_cutoff):
                        if target_vocab[topk_idxs[i]] == lemma:
                            probs[i] = 0
                    probs /= np.sum(probs)

                    new_samples = list(
                        np.random.choice(topk_idxs, self.config.n_represents * self.config.n_samples_per_rep,
                                         p=probs))

                    new_reps = []
                    for i in range(self.config.n_represents):
                        new_rep = {}
                        for j in range(self.config.n_samples_per_rep):
                            new_sample = target_vocab[new_samples.pop()]
                            new_rep[new_sample] = 1
                        new_reps.append(new_rep)
                    res[inst_id] = new_reps

            return res
        
    def get_sentence_tokens(self, sent):
        return [self.tokenizer.cls_token] + self.tokenizer.tokenize(sent) + [self.tokenizer.sep_token]

        
    def get_sentence_embedding(self, inst_id_to_sentence):
        res = {}
        with torch.no_grad():
            sorted_by_len = sorted(inst_id_to_sentence.items(), key=lambda x: len(x[1][0]) + len(x[1][2]))
            for batch in get_batch(sorted_by_len, 10):
                batch_sents = []
                for _, (pre, target, post) in batch:
                    tokens = self.get_sentence_tokens(pre + target + post)
                    batch_sents.append(tokens)
                
                tokens_idx = [self.tokenizer.convert_tokens_to_ids(sent_tokens) for sent_tokens in batch_sents]
                max_len = len(max(tokens_idx, key=lambda x: len(x)))
                
                input_batch = torch.zeros((len(tokens_idx), max_len), dtype=torch.long)
                for idx, row in enumerate(tokens_idx):
                    input_batch[idx, 0:len(row)] = torch.tensor(row)
                input_batch = input_batch.clone().detach().to(self.device)
                attention_mask = input_batch != 0
                sent_embeddings = self.model(input_batch, attention_mask=attention_mask).hidden_states[-1].mean(dim=1).detach().cpu().numpy()
                for (inst_id,( _, _, _)), sent_embedding in zip(batch, sent_embeddings):
                    res[inst_id] = sent_embedding

        return res
