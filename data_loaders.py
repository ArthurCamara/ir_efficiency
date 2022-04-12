from torch.utils.data import Dataset
import ir_datasets
import pytrec_eval
import random
import torch
from tqdm.auto import tqdm


class IRDatasetsLoader(Dataset):
    def __init__(self, tokenizer, qrels):
        self.data = ir_datasets.load("msmarco-document/train")
        self.tokenizer = tokenizer
        # Get all doc ids. Not optimal. But we are not measuring this here.
        self.all_doc_ids = [d.doc_id for d in tqdm(self.data.docs_iter(), total=3213835, leave=False)]
        # self.all_query_ids = [q.query_id for q in self.data.queries_iter()]
        self.train_qrels = pytrec_eval.parse_qrel(open(qrels))
        self.q_ids = dict(enumerate(self.train_qrels.keys()))

    def __getitem__(self, item):
        q_id = self.q_ids[item]
        d_id = list(self.train_qrels[q_id].keys())[0]
        neg_id = random.choice(self.all_doc_ids)
        if neg_id == d_id:
            neg_id = random.choice(self.all_doc_ids)
        pos_doc_obj = self.data.docs.lookup(d_id)
        pos_doc = f"{pos_doc_obj.url} {pos_doc_obj.title} {pos_doc_obj.body}"
        neg_doc_obj = self.data.docs.lookup(neg_id)
        neg_doc = f"{neg_doc_obj.url} {neg_doc_obj.title} {neg_doc_obj.body}"
        query_text = self.data.queries.lookup(q_id).text
        ret_dict = {
            "query_text": query_text,
            "doc_text": pos_doc,
            "neg_text": neg_doc,
        }
        return ret_dict

    def __len__(self):
        return len(self.q_ids)

    def cross_encoder_batcher(self, batch):
        texts = []
        labels = []
        for data in batch:
            texts.append([data["query_text"], data["doc_text"]])
            texts.append([data["query_text"], data["neg_text"]])
            labels.append(1.0)
            labels.append(0.0)

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        labels = torch.tensor(labels, dtype=torch.float)
        return tokenized, labels


class InMemoryLoader(Dataset):
    def __init__(self, tokenizer, docs_path, queries_path, qrels):
        self.tokenizer = tokenizer
        self.docs = {}
        self.queries = {}
        self.all_doc_ids = []
        # load all docs in memory
        for line in tqdm(open(docs_path), desc="reading docs", total=3213835, leave=False):
            d_id, doc = line.strip().split("\t", maxsplit=1)
            self.all_doc_ids.append(d_id)
            self.docs[d_id] = doc

        for line in tqdm(open(queries_path), desc="reading queries", total=367013, leave=False):
            d_id, doc = line.strip().split("\t", maxsplit=1)
            self.queries[d_id] = doc
        self.train_qrels = pytrec_eval.parse_qrel(open(qrels))
        self.q_ids = dict(enumerate(self.train_qrels.keys()))

    def __getitem__(self, item):
        q_id = self.q_ids[item]
        d_id = list(self.train_qrels[q_id].keys())[0]
        neg_id = random.choice(self.all_doc_ids)
        if neg_id == d_id:
            neg_id = random.choice(self.all_doc_ids)
        pos_doc = self.docs[d_id]
        neg_doc = self.docs[neg_id]
        query_text = self.queries[q_id]
        ret_dict = {
            "query_text": query_text,
            "doc_text": pos_doc,
            "neg_text": neg_doc,
        }
        return ret_dict

    def __len__(self):
        return len(self.q_ids)

    def cross_encoder_batcher(self, batch):
        texts = []
        labels = []
        for data in batch:
            texts.append([data["query_text"], data["doc_text"]])
            texts.append([data["query_text"], data["neg_text"]])
            labels.append(1.0)
            labels.append(0.0)

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        labels = torch.tensor(labels, dtype=torch.float)
        return tokenized, labels
