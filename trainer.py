import keyengine
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from transformers import AutoTokenizer

if __name__ == "__main__":
    print("Which would you like to train?")
    options = ["Which would you like to train?", "1. Encoder & Decoder"]

    option = keyengine.menu(options)
    while option == 0:
        option = keyengine.menu(options)

    if option == 1:
        print("Encoder & Decoder", end = '')
        input()
        input_size = input("Context Size: ") or 4096
        layers = input("Internal Layers: ") or 3
        bottleneck = input("Bottleneck Size: ") or 1024
        heads = input("Attention Heads: ") or 6

        encoder = utils.Encoder(int(input_size), int(heads), int(layers), int(bottleneck))
        decoder = utils.Decoder(int(input_size), int(layers), int(bottleneck))
        wrapper = utils.EncodeDecodeWrapper(encoder, decoder)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        optimizer = optim.Adam(wrapper.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        epochs = 1000

        data = []
        while True:
            i = input("Add Data: ")
            if not i:
                break
            data.append(tokenizer.encode(i))

        for epoch in tqdm.tqdm(range(epochs)):
            for i in data:
                optimizer.zero_grad()
                output = wrapper(torch.tensor(i))
                loss = criterion(output, torch.tensor(i))
                loss.backward()
                optimizer.step()

        torch.save(wrapper.encoder.state_dict(), "models/encoder.pth")
        torch.save(wrapper.decoder.state_dict(), "models/decoder.pth")
