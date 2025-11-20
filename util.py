import os
from threading import Thread
from tqdm import tqdm, trange
import numpy as np
from typing import Union, List, Tuple, Any
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="`max_length` is ignored when `padding`=`True` and there is no truncation strategy.")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def apk(actual, predicted, k=25):
    """
    Computes the average precision at k.
    
    This function computes the average prescision at k between two lists of
    items.
    
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]
        
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    """
    Computes the mean average precision at k.
    
    This function computes the mean average prescision at k between two lists
    of lists of items.
    
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """    
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def is_recall(x, y, topN=25):
    if y[0] in x[:topN]:
        return 1
    else:
        return 0


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def recall_inference(df, model, tokenizer, device, batch_size=16, max_length=512):
    pids = list(df['order_index'].values)
    sentences = list(df['query_text'].values)
    
    # 根据文本长度逆序：长的在前，短的在后
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    # print(length_sorted_idx[:5])
    # print(sentences_sorted[:5])
    
    all_embeddings = []
    for start_index in trange(0, len(sentences), batch_size, desc='Batches', disable=False):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]
        features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        # if start_index == 0:
        #     print(f'features = {features.keys()}')
        # features input_id, attention_mask
        features = batch_to_device(features, device)
        with torch.no_grad():
            outputs = model(**features)
            embeddings = last_token_pool(outputs.last_hidden_state, features['attention_mask'])
            # if start_index == 0:
            #     print(f'embeddings = {embeddings.detach().cpu().numpy().shape}')
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.detach().cpu().numpy().tolist()
        all_embeddings.extend(embeddings)
    
    all_embeddings = [np.array(all_embeddings[idx]).reshape(1, -1) for idx in np.argsort(length_sorted_idx)]
    
    sentence_embeddings = np.concatenate(all_embeddings, axis=0)
    result = {pids[i]: em for i, em in enumerate(sentence_embeddings)}
    return result


def remove_duplication(x, y):
    res = []
    for xi in x:
        if xi not in y:
            res.append(xi)
    return res


def recall_context(x, misconception_mapping_dic):
    res = []
    for xi in x:
        MisconceptionName = misconception_mapping_dic[xi]
        res.append({'MisconceptionName': MisconceptionName})
    return res


def recall_context_str(x, misconception_mapping_dic, topN):
    res = []
    for xi in x[:topN]:
        MisconceptionName = misconception_mapping_dic[xi]
        res.append(MisconceptionName)
    return res


def get_predict(df, query_embedding, sentence_embeddings, index_text_embeddings_index, device, RANK=100):
    predict_list = []
    for i, (_, row) in enumerate(df.iterrows()):
        query_id = row['order_index']
        query_em = query_embedding[query_id].reshape(1, -1)
        
        # torch
        # sentence_embeddings_tensor = torch.tensor(sentence_embeddings).to(device)
        # query_em = torch.tensor(query_em).to(device).view(1, -1)
        # score = (query_em @ sentence_embeddings_tensor.T).squeeze()
        # sort_index = torch.sort(-score).indices.detach().cpu().numpy().tolist()[:RANK]
        
        # numpy
        cosine_similarity = np.dot(query_em, sentence_embeddings.T).flatten()
        sort_index = np.argsort(-cosine_similarity)[:RANK]
        
        pids = [index_text_embeddings_index[index] for index in sort_index]
        predict_list.append(pids)
    return predict_list


def get_bge_recall(df, cnt=100):
    bge_recall = []
    for _, row in df.iterrows():
        query = row['query_text']
            
        # 正样本
        pos = []
        for data in row['new_positive_ctxs']:
            pos.append(data['MisconceptionName'])
        
        # 负样本
        neg = []
        hard_negative_ctxs = row['new_had_recall_ctxs'][:cnt]
        for data in hard_negative_ctxs:
            neg.append(data['MisconceptionName'])
        
        bge_recall.append({'query': query, 'pos': pos, 'neg': neg})
    return bge_recall
    

def get_bge_rank(df, cnt=50):
    bge_rank = []
    for _, row in df.iterrows():
        query = row['query_text']
        
        # 正样本
        pos = []
        for data in row['new_positive_ctxs']:
            pos.append(data['MisconceptionName'])
        
        # 负样本
        neg = []
        hard_negative_ctxs = row['new_had_recall_ctxs'][:cnt]
        for data in hard_negative_ctxs:
            neg.append(data['MisconceptionName'])
        
        bge_rank.append({
            'query': query,
            'pos': pos,
            'neg': neg,
            'prompt': "Given a query with a SubjectName, along with a ConstructName, QuestionText, CorrectAnswer, and Misconcepte Incorrect Answer, determine whether the Misconcepte Incorrect Answer is pertinent to the query by providing a prediction of either 'Yes' or 'No'."
        })
    return bge_rank


class DatasetForReranker(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer_path: str,
            max_len: int = 512,
            query_prefix: str = 'A: ',
            passage_prefix: str = 'B: ',
            cache_dir: str = None,
            prompt: str = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       trust_remote_code=True,
                                                       cache_dir=cache_dir)

        self.dataset = dataset
        self.max_len = max_len
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.total_len = len(self.dataset)

        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        self.prompt_inputs = self.tokenizer(prompt,
                                            return_tensors=None,
                                            add_special_tokens=False)['input_ids']
        sep = "\n"
        self.sep_inputs = self.tokenizer(sep,
                                         return_tensors=None,
                                         add_special_tokens=False)['input_ids']

        self.encode_max_length = self.max_len + len(self.sep_inputs) + len(self.prompt_inputs)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        query, passage = self.dataset[item]
        query = self.query_prefix + query
        passage = self.passage_prefix + passage
        query_inputs = self.tokenizer(query,
                                      return_tensors=None,
                                      add_special_tokens=False,
                                      max_length=self.max_len * 3 // 4,
                                      truncation=True)
        passage_inputs = self.tokenizer(passage,
                                        return_tensors=None,
                                        add_special_tokens=False,
                                        max_length=self.max_len,
                                        truncation=True)
        if self.tokenizer.bos_token_id is not None:
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                self.sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=self.encode_max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
        else:
            item = self.tokenizer.prepare_for_model(
                query_inputs['input_ids'],
                self.sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=self.encode_max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )

        item['input_ids'] = item['input_ids'] + self.sep_inputs + self.prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        item.pop('token_type_ids') if 'token_type_ids' in item.keys() else None
        if 'position_ids' in item.keys():
            item['position_ids'] = list(range(len(item['input_ids'])))

        return item


class collater():
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_multiple_of = 8
        self.label_pad_token_id = -100

    def __call__(self, data):
        labels = [feature["labels"] for feature in data] if "labels" in data[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in data:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        # print(data)
        return self.tokenizer.pad(
            data,
            padding=True,
            max_length=self.max_len,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )


def last_logit_pool(logits: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1, :]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i], :] for i in range(batch_size)], dim=0)


def last_logit_pool_layerwise(logits: Tensor,
                              attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FlagLLMReranker:
    def __init__(
            self,
            model_name_or_path: str = None,
            lora_name_or_path: str = None,
            use_fp16: bool = False,
            use_bf16: bool = False,
            cache_dir: str = None,
            device: Union[str, int] = None
    ) -> None:

        print("llm reranker:", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                  cache_dir=cache_dir,
                                                  trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eod_id is not None:
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token_id = tokenizer.im_start_id
                tokenizer.eos_token_id = tokenizer.im_end_id
        if 'mistral' in model_name_or_path.lower():
            print("left?")
            tokenizer.padding_side = 'left'

        self.tokenizer = tokenizer

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            # torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
            cache_dir=cache_dir,
            trust_remote_code=True,
            device_map=device)
        
        # model = PeftModel.from_pretrained(model, lora_name_or_path)
        # self.model = model.merge_and_unload()
        if lora_name_or_path != 'none':
            peft_config = LoraConfig(
                inference_mode=True,
                r=32,
                lora_alpha=64,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head'],
                lora_dropout=0.1,
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=None
            )
            model = get_peft_model(model, peft_config)
            d = torch.load(lora_name_or_path, map_location=model.device)
            model.load_state_dict(d, strict=False)
            self.model = model.merge_and_unload()
        
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.device = self.model.device
        # if use_fp16 and use_bf16 is False:
        #     self.model.half()

        self.model.eval()

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

    @torch.no_grad()
    def compute_score(self, sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], batch_size: int = 16,
                      max_length: int = 512, prompt: str = None, normalize: bool = False,
                      use_dataloader: bool = True, num_workers: int = None) -> List[float]:
        assert isinstance(sentence_pairs, list)

        # if self.num_gpus > 0:
        #     batch_size = batch_size * self.num_gpus

        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        length_sorted_idx = np.argsort([-self._text_length(q) - self._text_length(p) for q, p in sentence_pairs])
        sentences_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]

        dataset, dataloader = None, None
        if use_dataloader:
            if num_workers is None:
                num_workers = min(batch_size, 16)
            dataset = DatasetForReranker(sentences_sorted,
                                         self.model_name_or_path,
                                         max_length,
                                         cache_dir=self.cache_dir,
                                         prompt=prompt)
            dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=False,
                                    num_workers=num_workers,
                                    collate_fn=collater(self.tokenizer, max_length))

        all_scores = []
        if dataloader is not None:
            for inputs in tqdm(dataloader):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                scores = last_logit_pool(logits, inputs['attention_mask'])
                scores = scores[:, self.yes_loc]
                all_scores.extend(scores.cpu().float().tolist())
        else:
            if prompt is None:
                prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
            else:
                print(prompt)
            prompt_inputs = self.tokenizer(prompt,
                                           return_tensors=None,
                                           add_special_tokens=False)['input_ids']
            sep = "\n"
            sep_inputs = self.tokenizer(sep,
                                        return_tensors=None,
                                        add_special_tokens=False)['input_ids']
            encode_max_length = max_length * 2 + len(sep_inputs) + len(prompt_inputs)
            for batch_start in trange(0, len(sentences_sorted), batch_size):
                batch_sentences = sentences_sorted[batch_start:batch_start + batch_size]
                batch_sentences = [(f'A: {q}', f'B: {p}') for q, p in batch_sentences]
                queries = [s[0] for s in batch_sentences]
                passages = [s[1] for s in batch_sentences]
                queries_inputs = self.tokenizer(queries,
                                                return_tensors=None,
                                                add_special_tokens=False,
                                                max_length=max_length,
                                                truncation=True)
                passages_inputs = self.tokenizer(passages,
                                                 return_tensors=None,
                                                 add_special_tokens=False,
                                                 max_length=max_length,
                                                 truncation=True)

                batch_inputs = []
                for query_inputs, passage_inputs in zip(queries_inputs['input_ids'], passages_inputs['input_ids']):
                    item = self.tokenizer.prepare_for_model(
                        [self.tokenizer.bos_token_id] + query_inputs,
                        sep_inputs + passage_inputs,
                        truncation='only_second',
                        max_length=1024,
                        padding=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        add_special_tokens=False
                    )
                    item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
                    item['attention_mask'] = [1] * len(item['input_ids'])
                    item.pop('token_type_ids') if 'token_type_ids' in item.keys() else None
                    if 'position_ids' in item.keys():
                        item['position_ids'] = list(range(len(item['input_ids'])))
                    batch_inputs.append(item)
                    # print(query_inputs)
                    # print(passage_inputs)
                collater_instance = collater(self.tokenizer, max_length)
                batch_inputs = collater_instance(
                    [{'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']} for item in
                     batch_inputs])

                batch_inputs = {key: val.to(self.device) for key, val in batch_inputs.items()}
                # print(self.model.device)
                outputs = self.model(**batch_inputs, output_hidden_states=True)
                logits = outputs.logits
                scores = last_logit_pool(logits, batch_inputs['attention_mask'])
                scores = scores[:, self.yes_loc]
                all_scores.extend(scores.cpu().float().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        if len(all_scores) == 1:
            return all_scores[0]

        return all_scores

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings


def inference(df, model, prompt, batch_size=16, max_length=1024):
    scores = model.compute_score(list(df['predict_text'].values),
                                 batch_size=batch_size, max_length=max_length,
                                 use_dataloader=True, prompt=prompt)
    df['score'] = scores
    return df
