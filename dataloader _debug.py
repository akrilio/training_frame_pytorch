import torch
import torch.utils.data as Data

from model.datasets import BasicDataset
from image_searcher import Searcher


if __name__ == '__main__':
    searcher = Searcher(root_path=None, dataset_path=r'D:\Public\water_pipeline')
    lb_dataset, _ = searcher.read_dataset(label='all', equal=False, T=1)
    torch.multiprocessing.set_start_method('spawn', force=True)
    dataset = BasicDataset(lb_dataset['train'], (256, 512), 'strong')
    loader = Data.DataLoader(
        dataset=dataset, batch_size=128, shuffle=False, pin_memory=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2,
    )
    with torch.no_grad():
        for n, data in enumerate(loader, 1):
            x, y = data
            print(x.device, y.device)
            x = x.cuda()
            y = y.cuda()
            print(x.shape, y.shape)
            if n > 3:
                break
    print('finished')
