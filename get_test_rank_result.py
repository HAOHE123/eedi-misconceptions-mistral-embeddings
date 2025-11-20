import os
import sys
from threading import Thread
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Any
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from util import *
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


DEBUG = False
topN = 25
rank_batch_size = 16
rank_max_length = 1024
rank_model_path = sys.argv[1]
rank_lora_path = sys.argv[2]
prompt = "Given a query with a SubjectName, along with a ConstructName, QuestionText, CorrectAnswer, and Misconcepte Incorrect Answer, determine whether the Misconcepte Incorrect Answer is pertinent to the query by providing a prediction of either 'Yes' or 'No'."
data_path = '../input/eedi-mining-misconceptions-in-mathematics/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


misconception_mapping = pd.read_csv(data_path + 'misconception_mapping.csv')
misconception_mapping['query_text'] = misconception_mapping['MisconceptionName']
misconception_mapping['order_index'] = misconception_mapping['MisconceptionId']
misconception_mapping['MisconceptionId'] = misconception_mapping['MisconceptionId'].map(lambda x: int(x))
misconception_mapping_dic = misconception_mapping.set_index('MisconceptionId')['MisconceptionName'].to_dict()

dev_df = pd.read_csv('./output/valid.csv')
dev_df['top_recall_pids'] = dev_df['top_recall_pids'].map(eval)
dev_df['answer_id'] = dev_df['answer_id'].map(eval)
if DEBUG:
    dev_df = dev_df.head(30)


valid_mapk_score = mapk(list(dev_df['answer_id'].values), list(dev_df['top_recall_pids'].values), k=topN)
print(f'valid_mapk_score = {valid_mapk_score}')


dev_df['top_recall_text'] = dev_df['top_recall_pids'].apply(lambda x: recall_context_str(x, misconception_mapping_dic, topN))
    
dev_df['order_id'] = list(range(len(dev_df)))
predict_list = []
for _, row in dev_df.iterrows():
    for index, text in enumerate(row['top_recall_text'][:topN]):
        predict_list.append({
            "order_id": row['order_id'],
            'predict_text': [row['query_text'], text],
            'or_order': index,
            'candi_paper_id': row['top_recall_pids'][index],
            'answer_id': row['answer_id']
        })
predict_list = pd.DataFrame(predict_list)


model = FlagLLMReranker(
    rank_model_path,
    lora_name_or_path=rank_lora_path,
    use_fp16=True,
    device=device
)
predict_list = inference(predict_list, model, prompt, batch_size=rank_batch_size, max_length=rank_max_length)

predicts_test = []
answer_id_list = []
for _, df in predict_list.groupby('order_id'):
    scores = df['score'].values
    score_indexs = np.argsort(-scores)
    candi_paper_ids = df['candi_paper_id'].values
    predicts_test.append([candi_paper_ids[index] for index in score_indexs[:topN]])
    answer_id_list.append(list(df['answer_id'].values[0]))


valid_mapk_score = mapk(answer_id_list, predicts_test, k=topN)
print(f'valid_mapk_score = {valid_mapk_score}')
