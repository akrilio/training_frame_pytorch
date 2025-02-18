import torch
import numpy as np
from dataset_edit import Dataset_Editor
from model.datasets import ClassifyDataset
from model.vits import BasicViT


class Opener(BasicViT):
    def __init__(self, **params):
        super().__init__(**params)

    def return_input(self, x):
        b, n, c = x.shape
        x = x.reshape(b, 28, 28, 8, 8, -1)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(b, -1, 224, 224)
        return x

    def forward(self, x: torch.Tensor):
        eye = []
        x = self.in_shot(x)
        x = self.to_tokens(x)
        x = x + self.in_head(x)
        x += self.pos_embedding
        x = self.dropout(x)
        for block in self.transformers:
            x = block(x)
            eye.append(self.return_input(x.detach()))
        x = self.out_head(x)
        x = self.to_img(x)
        x = self.softmax(x)
        eye = torch.cat(eye, dim=0)
        return x, eye


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    params = {'image_size': (32, 32), 'channels': 3, 'classes': 10,
              'patch_size': (8, 8), 'depth': 12, 'dims': 384,
              'num_heads': 6, 'dim_heads': 64, 'mlp_ratio': 4,
              'dropout': 0.1, 'emb_dropout': 0.1}
    # model = BasicViT(**params)
    model_path = r'logs/BasicViT8-efficient-dropout-20240524-131152/last_epoch_81_train_loss_0.2670_val_loss0.8237.pt'
    # model.load_state_dict(weights, strict=False)
    weights = torch.load(model_path)
    for i in range(12):
        key_k = 'transformers.{}.attn.k_scale'.format(i)
        key_q = 'transformers.{}.attn.q_scale'.format(i)
        print('block: {}, \n\t k_scale: {}, q_scale: {}'.format(i, weights[key_k], weights[key_q]))
    """
    dataset_path = r'datasets/dataset_6_2_2'
    editor = Dataset_Editor(dataset_path)
    data = editor.read_data()
    dataset = ClassifyDataset(data['train'], input_shape=(224, 224), random=True)
    image_data, label_data = dataset.__getitem__(10)
    model = model.cuda()
    x, eye = model(image_data.unsqueeze(0))

    image = image_data.squeeze(0).cpu().detach().numpy()
    label = label_data.cpu().detach().numpy()
    fig, axes = plt.subplots(1, 6)
    axes[0].imshow(image)
    axes[0].axis('off')
    for l, ax in zip(label, axes[1:]):
        ax.imshow(l)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

    eye = eye.reshape(-1, 224, 224).cpu().detach().numpy()
    eye = ((eye - np.min(eye, axis=0, keepdims=True)) /
           (np.max(eye, axis=0, keepdims=True) - np.min(eye, axis=0, keepdims=True)))
    fig, axes = plt.subplots(6, 10)
    for e, ax in zip(eye, axes.flatten()):
        ax.imshow(e)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
"""