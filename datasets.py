"""Provides dataset classes for using with BEIR"""
from collections import defaultdict
from os import PathLike
from typing import Dict, List, Tuple, Type, Union

import pytrec_eval
import torch
import transformers
from transformers import AutoTokenizer
from sentence_transformers import InputExample
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from indexed_reader import IndexedReader
from utils import rawcount


# TODO: Expand to deal with robust's folds
class MsMarcoAxiomDataset(Dataset):
    def __init__(
        self,
        queries_file: PathLike,
        docs_file: Union[PathLike, IndexedReader],
        samples: PathLike,
        max_length: int = 512,
        dataset_name: str = "Msmarco",
        tokenizer: Type[transformers.PreTrainedTokenizer] = None,
        in_memory: bool = False,
    ) -> None:
        """Args:
        queries_file: PathLike object with a TSV with all queries
        docs_file: PathLike object with a TSV with all documents
        samples: PathLike object with a tsv with all triples (q_id, pos_id, neg_id)
        test: If True, will return the raw triples, so we can access them directly.
        triplets: Use triplets loss or ContrastiveLoss -> Hopefully get rid of it.
        in_memory: Keep data in memory or disk?
        """
        self.queries_dict = self.__load_tsv(queries_file)
        # Save some memory by re-using already created objects.
        if isinstance(docs_file, IndexedReader):
            self.docs = docs_file
        else:
            self.docs = IndexedReader(f"{dataset_name}-full", docs_file, in_memory=in_memory)

        self.samples = IndexedReader(samples.split("/")[-1], samples, index_as_id=True, simple_tsv=True)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index: int) -> InputExample:
        if index == len(self):
            raise StopIteration
        qid, pos_id, neg_id = self.samples.get_raw(index)[0].split("\t")
        query_text = self.queries_dict[qid]
        pos_text = self.docs[pos_id]
        neg_text = self.docs[neg_id]

        if self.tokenizer is None:
            return [query_text, pos_text, neg_text]
        # elif self.triplets:
        #     return InputExample(texts=[query_text, pos_text, neg_text])
        # If not triplets, return with labels.
        return (
            InputExample(texts=[query_text, pos_text], label=1),
            InputExample(texts=[query_text, neg_text], label=0),
        )

    def contrastive_batching(self, batch):
        """Receive a list of tuples to be flattened
        Returns two lists: One with all, flatened pairs, and one with all labels"""
        texts = []
        labels = []
        queries = []
        for pos_example, neg_example in batch:
            queries.append(pos_example.texts[0])
            queries.append(neg_example.texts[0])

            texts.append(pos_example.texts[1])
            texts.append(neg_example.texts[1])

            labels.append(pos_example.label)
            labels.append(neg_example.label)
        all_queries = self.tokenizer(
            queries, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        all_texts = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        return ((all_queries, all_texts), torch.tensor(labels, dtype=torch.int8))

    def cross_encoder_batcher(self, batch):
        texts = []
        labels = []

        for pos_example, neg_example in batch:
            texts.append([pos_example.texts[0], pos_example.texts[1]])
            texts.append([neg_example.texts[0], neg_example.texts[1]])

            labels.append(pos_example.label)
            labels.append(neg_example.label)

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        labels = torch.tensor(labels, dtype=torch.long)

        return tokenized, labels

    def text_encoder_batching(self, batch):
        tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
        return tokens

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def __load_tsv(tsv_path: PathLike) -> dict[str, str]:
        """Loads a TSV as dict"""
        d = {}
        file_name = tsv_path.split("/")[-1]
        for line in tqdm(open(tsv_path), desc=file_name, ncols=120, leave=False):
            _id, content = line.strip().split("\t", maxsplit=1)
            d[_id] = content
        return d

    def smart_batching(self, batch):
        all_queries = []
        all_positive = []
        all_negative = []

        for positive_pair, negative_pair in batch:
            all_queries.append(positive_pair.texts[0])
            all_positive.append(positive_pair.texts[1])
            all_negative.append(negative_pair.texts[1])

        all_queries = self.tokenizer(
            all_queries, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        all_positive = self.tokenizer(
            all_positive, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        all_negative = self.tokenizer(
            all_negative, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return (all_queries, all_positive, all_negative)


class MsMarcoRankerDataset(Dataset):
    """A Classs for a (re)-ranker dataset
    Args:
        queries_file: File with tsv-separeted queries
        docs_file: File with tsv-separeted docs OR a pre-computed IndexedReader
        run_file: A TREC-formatted run file for initial ranking (probably from BM25)
        tokenizer: A HuggingFace tokenizer object for tokenizing documents
        valid_queries: A list of query ids to filter from all the samples. Defaults to using all.
    """

    def __init__(
        self,
        queries_file: PathLike,
        docs_file: Union[PathLike, IndexedReader],
        run_file: PathLike,
        tokenizer: Type[transformers.PreTrainedTokenizer],
        max_length: int = 512,
        dataset_name: str = "Msmarco",
        valid_queries: List[str] = None,
    ):
        self.max_length = max_length
        self.queries_dict = self.__load_tsv(queries_file)
        if isinstance(docs_file, IndexedReader):
            self.docs = docs_file
        else:
            self.docs = IndexedReader(f"{dataset_name}-full", docs_file)
        self.samples = self.__load_run(run_file)
        self.tokenizer = tokenizer
        self.run = self._normalize_scores(pytrec_eval.parse_run(open(run_file)))
        self.current_index = None
        if valid_queries is not None:
            valid_queries = set(valid_queries)
            self.samples = [x for x in self.samples if x[0] in valid_queries]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, str, float]:
        if index == len(self):
            raise StopIteration

        query_id, doc_id, _ = self.samples[index]

        query_text = self.queries_dict[query_id]
        doc_text = self.docs[doc_id]
        ret_dict = {"query_id": query_id, "doc_id": doc_id, "query_text": query_text, "doc_text": doc_text}

        return ret_dict

    def smart_batching(self, batch):
        """Use this function when batching i.e.:
        dataloader = Dataloader(dataset)
        dataloader.collate_fn = dataset.smart_batching
        """
        all_queries = []
        all_docs = []
        all_doc_ids = []
        all_query_ids = []
        for data in batch:
            all_queries.append(data["query_text"])
            all_docs.append(data["doc_text"])
            all_query_ids.append(data["query_id"])
            all_doc_ids.append(data["doc_id"])

        all_queries = self.tokenizer(
            all_queries, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
        )
        all_docs = self.tokenizer(
            all_docs, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
        )
        return {
            "all_queries": all_queries,
            "all_docs": all_docs,
            "all_query_ids": all_query_ids,
            "all_doc_ids": all_doc_ids,
        }

    def cross_encoder_batcher(self, batch):
        texts = []
        all_doc_ids = []
        all_query_ids = []
        for data in batch:
            texts.append([data["query_text"], data["doc_text"]])
            all_query_ids.append(data["query_id"])
            all_doc_ids.append(data["doc_id"])

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return {"tokens": tokenized, "all_query_ids": all_query_ids, "all_doc_ids": all_doc_ids}

    @staticmethod
    def __load_tsv(tsv_path: PathLike) -> dict[str, str]:
        """Loads a TSV as dict"""
        d = {}
        file_name = tsv_path.split("/")[-1]
        for line in tqdm(open(tsv_path), desc=file_name, ncols=120, leave=False):
            _id, content = line.strip().split("\t", maxsplit=1)
            d[_id] = content
        return d

    def __load_run(self, run_path: PathLike, show_pbar: bool = False) -> List[Tuple[str, str, float]]:
        """Loads a TREC-formatted run file, return as list of tuples
        Args:
            run_path: Path with TREC-formatted run
            show_pbar: boolean flag for showing progressbar when reading file
        Return:
            List of tuples with (query_id, doc_id, score)
        """
        n_lines = rawcount(run_path)
        samples = []
        for line in tqdm(open(run_path), total=n_lines, ncols=120, desc="Run_file", leave=False, disable=not show_pbar):
            q_id, _, d_id, _, score, _ = line.split()
            samples.append((q_id, d_id, float(score)))
        return samples

    @staticmethod
    def _normalize_scores(run: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Normalize scores from a run, from 0 to 1"""
        return_dict = defaultdict(lambda: dict())
        for qid in run:
            all_values = list(run[qid].values())
            min_val = min(all_values)
            max_val = max(all_values)
            for did, score in run[qid].items():
                return_dict[qid][did] = (score - min_val) / (max_val - min_val)

        return dict(return_dict)

    def get_score(self, query_id: str, doc_id: str) -> float:
        """Returns the score of a given document from a given query_id, from a normalized run score"""
        return self.run[query_id][doc_id]


# FROM SPLADE
class CollectionDatasetPreLoad:
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    """

    def __init__(self, data_path):
        self.data_path = data_path

        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        self.line_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        print("Preloading dataset", flush=True)
        with open(self.data_path) as reader:
            for i, line in enumerate(tqdm(reader, total=rawcount((data_path)))):
                if len(line) > 1:
                    id_, *data = line.split("\t")  # first column is id
                    data = " ".join(" ".join(data).splitlines())
                    self.data_dict[i] = data
                    self.line_dict[i] = id_.strip()

        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        return self.line_dict[idx], self.data_dict[idx]


class DataLoaderSplade(torch.utils.data.dataloader.DataLoader):
    def __init__(self, tokenizer_path, max_length, **kwargs):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        super().__init__(collate_fn=self.collate_fn, **kwargs, pin_memory=True)

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(
            list(d),
            add_special_tokens=True,
            padding="longest",  # pad to max sequence length in batch
            truncation="longest_first",  # truncates to max model length,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        return {
            **{k: torch.tensor(v) for k, v in processed_passage.items()},
            "id": [i for i in id_],
            "text": d,
        }
