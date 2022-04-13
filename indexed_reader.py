"""Defines a class that reads and writes to files using an indexed mapping in-memory"""
import logging
import os
import pickle
import mmap
from typing import Optional, Union

from tqdm.auto import tqdm

from utils import g_path, rawcount

DATA_HOME = "./data/"


class IndexedReader:
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        index_as_id: bool = False,
        cache_path: str = "cache",
        force: bool = False,
        n_lines: Optional[int] = None,
        split_str: str = "\t",
    ):
        """Defines a class for dealing with indexed files
        Args:
            dataset_name: String with name of the dataset to be used.
                Index file will be <dataset_name>_index.pkl and data will be <dataset_name>.txt
            dataset_path: Path with tsv files to be used to load data.
            cache_path: Path to store files on disk. Defaults to DATA_HOME/cache
            forced: Boolean to force to re-compute the index
            n_lines: Optional for number of lines to be read from file, for a prettier progress bar
            split_str: Sting to be used as termination for spliiting the object. Defaults to tab
        """

        self.current_index = 0
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.index_as_id = index_as_id
        self.cache_path = g_path(DATA_HOME, cache_path)
        self.from_redis = False
        self.force = force
        self.n_lines = n_lines
        self.split_str = split_str

        if not self.n_lines:
            self.n_lines = rawcount(dataset_path)
        self._create_index()

    def _create_index(self):
        """Create in-memory index for the reader file and Saves a copy to disk.

        The index object (self.index[]) is a dictionary that maps, for each item_id `id`,
        the position where the attributes for `id` start on the target file
        (self.dataset_path) and how many lines are expected for that entity. Example:
        self.index["<http://dbpedia.org/resource/An_American_in_Paris>"] = (1234, 5) means that,
        for the document "<http://dbpedia.org/resource/An_American_in_Paris>", starting on position number
        1234 (in bytes) of file self.dataset_path, there are 5 lines of attributes that are mapped to e.
        This greatly decreases memory usage and makes it faster to access data on disk, specially using SSDs.
        """
        index_file = g_path(self.cache_path, f"{self.dataset_name}_index.pkl")
        if os.path.isfile(index_file) and not self.force:
            logging.info(
                f"Already found processed dataset {self.dataset_name}. Refusing to re-create. Will only load the index"
            )
            self.index = pickle.load(open(index_file, "rb"))
            self.f = open(self.dataset_path, "r+b")
            self.reader = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
            self.all_ids = list(self.index.keys())

            return
        self.index = {}
        current_doc = None
        f = open(self.dataset_path, "r+b")
        m = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        m.madvise(mmap.MADV_WILLNEED)
        current_doc_count = 0
        position = f.tell()
        start_position = position
        line = m.readline()
        read_lines = 0
        pbar = tqdm(desc=f"Loading {self.dataset_name}", ncols=120, total=self.n_lines)
        while line != b"":  # While not at EOF
            read_lines += 1
            pbar.update()
            c_line = line.decode("utf-8").strip()
            if c_line[0] == "#":  # Comment
                position = m.tell()
                line = m.readline()
                continue
            try:
                s, _ = c_line.split(maxsplit=1)
            except ValueError:  # Empty Document
                s = c_line.strip()

            if (s != current_doc and current_doc is not None) or self.index_as_id:  # New entity or index as id
                if self.index_as_id:
                    self.index[read_lines - 1] = (start_position, 1)
                else:
                    self.index[current_doc] = (start_position, current_doc_count)
                start_position = position
                current_doc_count = 0
            current_doc_count += 1
            current_doc = s
            position = m.tell()
            line = m.readline()
        # Done
        if self.index_as_id:
            self.index[read_lines] = (start_position, 1)
        else:
            self.index[current_doc] = (start_position, current_doc_count)
        self.f = f
        self.reader = m
        self.reader.madvise(mmap.MADV_RANDOM)
        self.reader.madvise(mmap.MADV_WILLNEED)
        pbar.close()
        pickle.dump(self.index, open(index_file, "wb"))
        self.all_ids = list(self.index.keys())

    def __getitem__(self, object_id: Union[str, int]) -> str:
        """Gets an item from the dataset, either in memory or from a file
        Args:
            object_id: either an int or string representing the id of the requested object
        Returns:
            A string with the full document content, and new lines replaced as spaces.
        """
        if object_id not in self.index:
            return ""

        start_position, n_lines = self.index[object_id]
        self.reader.seek(start_position)

        lines = []

        for idx in range(n_lines):
            c_line = self.reader.readline().decode("utf-8").strip()
            if c_line == "#":
                continue
            if idx == 0:
                try:
                    lines.append(c_line.split("\t", maxsplit=1)[1])
                except IndexError:
                    lines.append(" ")
            else:
                lines.append(c_line)

            # This works here because we actually one have one line. But beware it may break on othere scenarios!
            return lines[0]

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        doc_id = self.all_ids[self.current_index]
        self.current_index += 1
        return self[doc_id]
