import math
import random
import torch

import numpy as np

from threading import current_thread


class Dataset:
    def __init__(self, samples, modelF, WT, args) -> None:
        # modelF is an additional argument 
        self.samples = samples
        self.modelF = modelF
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(self.samples) / args.batch_size)
        self.speaker_to_idx = {"M": 0, "F": 1}
        self.modalities = args.modalities
        self.dataset = args.dataset
        if WT:
            self.modelF.train()
        else:
            self.modelF.eval()

        # self.embedding_dim = args.dataset_embedding_dims[args.dataset][args.modalities]
        self.embedding_dim= 1024

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size : (index + 1) * self.batch_size]

        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s.text) for s in samples]).long()
        # I guess we choose max= mx as the maximum length dialogue in 
        # terms of utterances in our current batch, and use that as padding. 
        mx = torch.max(text_len_tensor).item()

        input_tensor = torch.zeros((batch_size, mx, self.embedding_dim))
        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        utterances = []
        for i, s in enumerate(samples):
            cur_len = len(s.text)
            utterances.append(s.sentence)
            tmp = []
            losst = 0
            for t, a, v in zip(s.sbert_sentence_embeddings, s.audio, s.visual):
                t = torch.tensor(t, dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.float32)
                v = torch.tensor(v, dtype=torch.float32)
                # Instead of using simple concatanation(as torch.cat(a, v) like COGMEN) a fusion module is used(self.modelF())
                # this fusion module projects all modalities into a shared feature space from which we get input vectors. 
                if self.modalities == "atv":
                    output, loss = self.modelF(a, t, v)
                    # print(output.shape)
                    tmp.append(output)
                    losst += loss
                elif self.modalities == "at":
                    output, loss = self.modelF(a, t, None)
                    # print(output.shape)
                    tmp.append(output)
                    losst += loss
                elif self.modalities == "tv":
                    output, loss = self.modelF(None, t, v)
                    tmp.append(output)
                    losst += loss
                elif self.modalities == "av":
                    output, loss = self.modelF(a, None, v)
                    tmp.append(output)
                    losst += loss
                elif self.modalities == "a":
                    output, loss = self.modelF(a, None, None)
                    tmp.append(output.squeeze(0))
                    losst += loss
                elif self.modalities == "t":
                    output, loss = self.modelF(None, t, None)
                    tmp.append(output.squeeze(0))
                    losst += loss
                elif self.modalities == "v":
                    output, loss = self.modelF(None, None, v)
                    tmp.append(output.squeeze(0))
                    losst += loss

            tmp = torch.stack(tmp)
            # print("Hoi", input_tensor.shape)
            # [12, 66, 1024]): (batch_size, mx, self.embedding_dim) 
            # first two are variable while training. 
            input_tensor[i, :cur_len, :] = tmp
            if self.dataset in ["meld", "dailydialog"]:
                # embed = torch.argmax(torch.tensor(s.speaker), dim=0)
                # why is it doing an argmax here? 
                speaker_tensor[i, :cur_len] = torch.tensor([s.speaker])
                # speaker_tensor[i, :cur_len] = embed
            else:
                speaker_tensor[i, :cur_len] = torch.tensor(
                    [self.speaker_to_idx[c] for c in s.speaker]
                )

            labels.extend(s.label)

        label_tensor = torch.tensor(labels).long()
        data = {
            "text_len_tensor": text_len_tensor, # this is essentially lengths of each dialogue in a batch. 
            "input_tensor": input_tensor, # I suppose this is the input to the transformer layer post fusion, 
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "utterance_texts": utterances,
            "encoder_loss": losst # extra element thats not in COGMEN
        }
        return data

    def shuffle(self):
        random.shuffle(self.samples)
