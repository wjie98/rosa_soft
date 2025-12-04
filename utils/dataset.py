import os
import torch
from datasets import load_from_disk, Dataset

__all__ = [
    "PackedPretrainDataset",
]


class PackedPretrainDataset(torch.utils.data.IterableDataset):
    def __init__(self,
        dataset: Dataset,
        block_size: int = 2048,
        tokenizer = None,
    ):
        dataset.num_rows
        self.dataset = dataset
        self.block_size = block_size
        self.tokenizer = tokenizer
    
    def __iter__(self):
        yield from self.__call__()
    
    def __call__(self, shuffle: bool = True):
        buffer = []
        while True:
            dataset = self.dataset
            if shuffle:
                dataset = dataset.shuffle()

            for item in dataset:
                if "input_ids" in item:
                    buffer.extend(item["input_ids"])
                elif "text" in item:
                    assert self.tokenizer is not None
                    tokenized = self.tokenizer(item["text"], truncation=False, padding=False)
                    buffer.extend(tokenized["input_ids"])
                else:
                    raise RuntimeError
                
                if len(buffer) >= self.block_size:
                    chunk = buffer[:self.block_size]
                    buffer = buffer[self.block_size:]
                    
                    input_ids = torch.tensor(chunk, dtype=torch.long)

                    yield {
                        "input_ids": input_ids,
                        "labels": input_ids.clone(),
                        # "attention_mask": torch.ones(self.seq_length, dtype=torch.bool)
                    }
    
    def loader(self, batch_size: int, num_workers: int = 0, pin_memory: bool = True):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)


if __name__ == "__main__":
    dataset = load_from_disk(os.path.join("~/DATA/llm_data/rosa_pretrain_small"))
    dataset = PackedPretrainDataset(dataset["train"], block_size=512)
    loader = dataset.loader(8, num_workers=0)
    
    # print(len(loader))

    for item in loader:
        print(item)
        break
    