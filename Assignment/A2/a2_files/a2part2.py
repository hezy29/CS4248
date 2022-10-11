import os
import re
import sys
import string
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


class LangDataset(Dataset):
    """
    Define a pytorch dataset class that accepts a text path, and optionally label path and
    a vocabulary (depends on your implementation). This class holds all the data and implement
    a __getitem__ method to be used by a Python generator object or other classes that need it.

    DO NOT shuffle the dataset here, and DO NOT pad the tensor here.
    """

    def __init__(self, text_path, label_path=None, vocab=None):
        """
        Read the content of vocab and text_file
        Args:
            vocab (string): Path to the vocabulary file.
            text_file (string): Path to the text file.
        """
        self.texts = None
        self.labels = None

        with open(text_path) as f:
            self.texts = [x[:-1] for x in f.readlines()]
        f.close()
        if not label_path == None:
            with open(label_path) as f:
                self.labels = [x[:-1] for x in f.readlines()]
            f.close()

        if not vocab:
            characters = {}
            idx = 0
            for text in self.texts:
                char_set = set(
                    [
                        x + y
                        for x, y in zip(
                            re.sub(r"[^\w\s]", "", text)[:-1],
                            re.sub(r"[^\w\s]", "", text)[1:],
                        )
                    ]
                )
                for each_char in char_set:
                    if not each_char in characters:
                        characters[each_char] = idx
                        idx += 1
            self.vocab = characters

    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
            num_class: number of class labels
        """
        num_vocab = len(self.vocab)
        num_class = len(set(self.labels))
        return num_vocab, num_class

    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.texts)

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).

        DO NOT pad the tensor here, do it at the collator function.
        """
        # embedding_dim = 16
        # bigram_embeds = nn.Embedding(
        #     num_embeddings=len(self.vocab), embedding_dim=embedding_dim
        # )
        # labels_idx = {
        #     x: i for x, i in zip(set(self.labels), range(len(set(self.labels))))
        # }
        # label_embeds = nn.Embedding(
        #     num_embeddings=len(set(self.labels)), embedding_dim=embedding_dim
        # )
        # text = (
        #     torch.stack(
        #         [
        #             bigram_embeds(torch.tensor([self.vocab[x]], dtype=torch.long))
        #             for x in set(
        #                 [
        #                     x + y
        #                     for x, y in zip(
        #                         re.sub(r"[^\w\s]", "", self.texts[i])[:-1],
        #                         re.sub(r"[^\w\s]", "", self.texts[i])[1:],
        #                     )
        #                 ]
        #             )
        #         ]
        #     )
        #     .mean(axis=0)
        #     .reshape(-1)
        # )
        # label = label_embeds(
        #     torch.tensor([labels_idx[self.labels[i]]], dtype=torch.long)
        # )
        tokens = re.sub(r"[^\w\s]", "", self.texts[i])
        text = [self.vocab[x + y] for x, y in zip(tokens[:-1], tokens[1:])]
        label = self.labels[i]

        return text, label


class Model(nn.Module):
    """
    Define a model that with one embedding layer with dimension 16 and
    a feed-forward layers that reduce the dimension from 16 to 200 with ReLU activation
    a dropout layer, and a feed-forward layers that reduce the dimension from 200 to num_class
    """

    def __init__(self, num_vocab, num_class, dropout=0.3):
        super().__init__()
        # define your model here
        self.linear1 = nn.Linear(num_vocab, 200)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(200, num_class)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # define the forward function here
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x


def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    labels_idx = {"eng": 0, "deu": 1, "fra": 2, "ita": 3, "spa": 4}
    unit_text, unit_label = [], []
    for unit in batch:
        unpadded = torch.tensor(unit[0])
        padded = F.pad(unpadded, (0, max([len(x[0]) for x in batch]) - len(unpadded)))
        unit_text.append(padded)
        unit_label.append(labels_idx[unit[1]])
    texts = torch.stack(unit_text)
    labels = torch.tensor(unit_label)

    return texts, labels


def train(
    model, dataset, batch_size, learning_rate, num_epoch, device="cpu", model_path=None
):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.

    Do not calculate the loss from padding.
    """
    data_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator, shuffle=True
    )

    # assign these variables
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    embedding_dim = 16

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a tuple of (inputs, labels)
            texts = data[0].to(device)
            labels = data[1].to(device)

            bigram_embeds = nn.Embedding(
                num_embeddings=len(dataset.vocab), embedding_dim=embedding_dim
            )

            labels_idx = {"eng": 0, "deu": 1, "fra": 2, "ita": 3, "spa": 4}
            label_embeds = nn.Embedding(
                num_embeddings=len(labels_idx), embedding_dim=embedding_dim
            )

            x = []

            for i in range(texts.size(0)):
                text = texts[i, :].tolist()
                print(dataset.vocab)
                text_embed = (
                    torch.stack(
                        [
                            bigram_embeds(
                                torch.tensor(
                                    [dataset.vocab[bigram_idx]], dtype=torch.long
                                )
                            )
                            for bigram_idx in text
                        ]
                    )
                    .mean(axis=0)
                    .reshape(-1)
                )
                print(text_embed)
                x.append(text_embed)

            x = torch.stack(x)

            y = torch.stack(
                torch.tensor(
                    [
                        label_embeds(
                            torch.tensor([labels_idx[label]], dtype=torch.long)
                        )
                        for label in labels
                    ]
                )
            )

            # zero the parameter gradients
            for param in model.parameters():
                param.grad = None

            # do forward propagation
            y_pred = model(x)

            # do loss calculation
            loss = criterion(y, y_pred)

            # do backward propagation
            loss.backward()

            # do parameter optimization step
            optimizer.step()

            # calculate running loss value for non padding

            # print loss value every 100 steps and reset the running loss
            if step % 100 == 99:
                print(
                    "[%d, %5d] loss: %.3f" % (epoch + 1, step + 1, running_loss / 100)
                )
                running_loss = 0.0

    end = datetime.datetime.now()

    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = None
    torch.save(checkpoint, model_path)

    print("Model saved in ", model_path)
    print("Training finished in {} minutes.".format((end - start).seconds / 60.0))


def test(model, dataset, class_map, device="cpu"):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    labels = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            outputs = model(texts).cpu()
            # get the label predictions

    return labels


def main(args):
    if torch.cuda.is_available():
        device_str = "cuda:{}".format(0)
    else:
        device_str = "cpu"
    device = torch.device(device_str)

    assert args.train or args.test, "Please specify --train or --test"
    if args.train:
        assert (
            args.label_path is not None
        ), "Please provide the labels for training using --label_path argument"
        dataset = LangDataset(args.text_path, args.label_path)
        num_vocab, num_class = dataset.vocab_size()
        model = Model(num_vocab, num_class).to(device)

        # you may change these hyper-parameters
        learning_rate = 0.01
        batch_size = 10
        num_epochs = 100

        train(
            model,
            dataset,
            batch_size,
            learning_rate,
            num_epochs,
            device,
            args.model_path,
        )
    if args.test:
        assert (
            args.model_path is not None
        ), "Please provide the model to test using --model_path argument"

        # create the test dataset object using LangDataset class

        # initialize and load the model

        # the lang map should contain the mapping between class id to the language id (e.g. eng, fra, etc.)
        lang_map = None

        # run the prediction
        preds = test(model, dataset, lang_map, device)

        # write the output
        with open(args.output_path, "w", encoding="utf-8") as out:
            out.write("\n".join(preds))
    print("\n==== A2 Part 2 Done ====")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", help="path to the text file")
    parser.add_argument("--label_path", default=None, help="path to the label file")
    parser.add_argument(
        "--train", default=False, action="store_true", help="train the model"
    )
    parser.add_argument(
        "--test", default=False, action="store_true", help="test the model"
    )
    parser.add_argument(
        "--model_path", required=True, help="path to the output file during testing"
    )
    parser.add_argument(
        "--output_path",
        default="out.txt",
        help="path to the output file during testing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
