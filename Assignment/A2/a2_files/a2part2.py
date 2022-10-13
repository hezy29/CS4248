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
        self.vocab = vocab  # initialize vocab for test procedure

        # Loading texts
        with open(text_path) as f:
            self.texts = [x[:-1] for x in f.readlines()]
        f.close()

        # Loading labels (for training)
        if not label_path == None:
            with open(label_path) as f:
                self.labels = [x[:-1] for x in f.readlines()]
            f.close()

        # Constructing bigram index (for training)
        if not vocab:
            characters = {}
            idx = 1
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
        num_class = None
        if not self.labels == None:
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
        # Tokenizing
        tokens = re.sub(r"[^\w\s]", "", self.texts[i])

        # Constructing character bigram
        text = [
            self.vocab[x + y]
            if x + y in self.vocab
            else 0  # OOV words to be considered as padding item
            for x, y in zip(tokens[:-1], tokens[1:])
        ]

        label = None
        # Getting label (for training)
        if not self.labels == None:
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
        self.embedding = nn.Embedding(
            num_embeddings=num_vocab + 1, embedding_dim=16, padding_idx=0
        )
        self.fc1 = nn.Linear(16, 200)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(200, num_class)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # define the forward function here
        x_non_padding = x.count_nonzero(
            dim=1
        )  # Record non-padding item number for each text
        x_non_padding = x_non_padding.reshape(x_non_padding.size(0), 1)
        x = self.embedding(x)  # Bigram Embedding for each text
        x = (x.sum(dim=1)) / x_non_padding  # Text Embedding ignoring padding
        x = nn.functional.normalize(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    # Label index
    labels_idx = {"eng": 0, "deu": 1, "fra": 2, "ita": 3, "spa": 4}
    unit_text, unit_label = [], []
    for unit in batch:
        unpadded = torch.tensor(unit[0])
        padded = F.pad(unpadded, (0, max([len(x[0]) for x in batch]) - len(unpadded)))
        unit_text.append(padded)

        if not unit[1]:  # Testing without labels input
            continue
        unit_label.append(labels_idx[unit[1]])
    texts = torch.stack(unit_text)
    if not unit_label:  # Testing without labels input
        labels = unit_label
    else:
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)  # Optimizing using ADAM algorithm with 1e-3 weight_decay to prevent overfitting

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a tuple of (inputs, labels)
            texts = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            for param in model.parameters():
                param.grad = None

            # do forward propagation
            labels_pred = model(texts)

            # do loss calculation
            loss = criterion(labels_pred, labels)

            # do backward propagation
            loss.backward()

            # do parameter optimization step
            optimizer.step()

            # calculate running loss value for non padding
            running_loss += loss.item()

            # print loss value every 100 steps and reset the running loss
            if step % 100 == 99:
                print(
                    "[%d, %5d] loss: %.3f" % (epoch + 1, step + 1, running_loss / 100)
                )
                running_loss = 0.0

    end = datetime.datetime.now()

    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = {
        "state_dict": model.state_dict(),
        "params": dataset.vocab_size(),
        "vocab": dataset.vocab,
    }
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
            outputs = model(texts)
            # get the label predictions
            labels += [class_map[idx] for idx in outputs.argmax(dim=1).tolist()]

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
        batch_size = 20
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
        trained = torch.load(args.model_path)
        dataset = LangDataset(args.text_path, vocab=trained["vocab"])
        num_vocab, num_class = dataset.vocab_size()

        # initialize and load the model
        model = Model(trained["params"][0], trained["params"][1]).to(device)
        model.load_state_dict(trained["state_dict"])

        # the lang map should contain the mapping between class id to the language id (e.g. eng, fra, etc.)
        lang_map = {
            k: v for v, k in {"eng": 0, "deu": 1, "fra": 2, "ita": 3, "spa": 4}.items()
        }

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
