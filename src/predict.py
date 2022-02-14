# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.predict import AudioPredictDataset
from typing import TypeVar, Any

T_co = TypeVar('T_co', covariant=True)
from src.model import create_model
import torch
from DeepLearningTemplate.data_preparation import parse_transforms, AudioLoader
from os.path import isfile
from torch.utils.data import DataLoader
from tqdm import tqdm


#def
def collate_fn(batch):
    batch_of_sample = []
    batch_of_n_splits = []
    batch_of_target_length = []
    for v in batch:
        batch_of_sample.append(v[0])
        batch_of_n_splits.append(v[1])
        batch_of_target_length.append(v[2])
    batch_of_sample = torch.cat(batch_of_sample)
    batch_of_n_splits = torch.tensor(batch_of_n_splits)
    batch_of_target_length = torch.tensor(batch_of_target_length)
    return batch_of_sample, batch_of_n_splits, batch_of_target_length


# class
class AudioPredictDataset(AudioPredictDataset):
    def __init__(self,
                 root,
                 loader,
                 transform,
                 max_waveform_length,
                 extensions=('.wav', '.flac')) -> None:
        super().__init__(root, loader, transform, extensions)
        self.max_waveform_length = max_waveform_length

    def __getitem__(self, index) -> T_co:
        path = self.samples[index]
        sample = self.get_sample(path=path)
        _, target_length = sample.shape
        temp = []
        for idx in range(0, target_length, self.max_waveform_length):
            s = self.transform(sample[:, idx:idx +
                                      self.max_waveform_length])[None]
            temp.append(s)
        # the transformed sample dimension is (n_splits, in_chans, length)
        sample = torch.cat(temp, 0)
        n_splits = sample.shape[0]
        return sample, n_splits, target_length


class Predict:
    def __init__(self, project_parameters) -> None:
        self.model = create_model(project_parameters=project_parameters).eval()
        if project_parameters.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.device = project_parameters.device
        self.batch_size = project_parameters.batch_size
        self.num_workers = project_parameters.num_workers
        self.classes = project_parameters.classes
        self.loader = AudioLoader(sample_rate=project_parameters.sample_rate)
        self.max_waveform_length = project_parameters.max_waveform_length
        self.in_chans = project_parameters.in_chans

    def predict(self, inputs) -> Any:
        if isfile(path=inputs):
            # predict the file
            sample = self.loader(path=inputs)
            in_chans, _ = sample.shape
            if in_chans != self.in_chans:
                sample = sample.mean(0)
                sample = torch.cat(
                    [sample[None] for idx in range(self.in_chans)])
            _, target_length = sample.shape
            temp = []
            for idx in range(0, target_length, self.max_waveform_length):
                s = self.transform(sample[:, idx:idx +
                                          self.max_waveform_length])[None]
                temp.append(s)
            # the transformed sample dimension is (n_splits, in_chans, length)
            sample = torch.cat(temp, 0)
            if self.device == 'cuda' and torch.cuda.is_available():
                sample = sample.cuda()
            with torch.no_grad():
                sample_hat = self.model(sample)
                sample_hat = torch.cat([v for v in sample_hat], -1)
                sample_hat = sample_hat[..., :target_length]
                return sample_hat
        else:
            result = []
            # predict the file from folder
            dataset = AudioPredictDataset(
                root=inputs,
                loader=self.loader,
                transform=self.transform,
                max_waveform_length=self.max_waveform_length)
            pin_memory = True if self.device == 'cuda' and torch.cuda.is_available(
            ) else False
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=pin_memory,
                                     collate_fn=collate_fn)
            with torch.no_grad():
                for sample, n_splits, target_length in tqdm(data_loader):
                    if self.device == 'cuda' and torch.cuda.is_available():
                        sample = sample.cuda()
                    sample_hat = self.model(sample)
                    temp = []
                    idx = 0
                    for ns, t in zip(n_splits, target_length):
                        s = torch.cat([v for v in sample_hat[idx:idx + ns]],
                                      -1)[..., :t]
                        temp.append(s)
                    result += temp
            return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict file
    result = Predict(project_parameters=project_parameters).predict(
        inputs=project_parameters.root)
