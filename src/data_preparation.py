# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.data_preparation import MySPEECHCOMMANDS, MyAudioFolder, AudioLightningDataModule
from typing import Union, Optional, Tuple, Callable, Any
from pathlib import Path
from torch import Tensor
import torch
from glob import glob
from os.path import join


#def
def add_noise_to_signal(signal, snr):
    sig_avg_watts = torch.mean(signal)
    sig_avg_db = 10 * torch.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - snr
    noise_avg_watts = 10**(noise_avg_db / 10)
    if torch.isnan(noise_avg_watts):
        noise_avg_watts = torch.tensor(0.01)
    noise = torch.normal(mean=0,
                         std=torch.sqrt(noise_avg_watts),
                         size=signal.shape)
    return signal + noise


def create_datamodule(project_parameters):
    if project_parameters.predefined_dataset:
        dataset_class = eval('My{}'.format(
            project_parameters.predefined_dataset))
    else:
        dataset_class = MyAudioFolder
    return AudioLightningDataModule(
        root=project_parameters.root,
        predefined_dataset=project_parameters.predefined_dataset,
        classes=project_parameters.classes,
        max_samples=project_parameters.max_samples,
        batch_size=project_parameters.batch_size,
        num_workers=project_parameters.num_workers,
        device=project_parameters.device,
        transforms_config=project_parameters.transforms_config,
        target_transforms_config=project_parameters.target_transforms_config,
        sample_rate=project_parameters.sample_rate,
        dataset_class=dataset_class)


#class
class MySPEECHCOMMANDS(MySPEECHCOMMANDS):
    def __init__(self,
                 root: Union[str, Path],
                 loader,
                 transform,
                 target_transform,
                 download: bool = False,
                 subset: Optional[str] = None) -> None:
        super().__init__(root, loader, transform, target_transform, download,
                         subset)
        self.classes = ['clean', 'mixed']
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        path = self._walker[n]
        clean = self.loader(path)
        #add whitenoise
        mixed = add_noise_to_signal(signal=clean, snr=0)
        if self.transform is not None:
            mixed = self.transform(mixed)
        if self.target_transform is not None:
            clean = self.target_transform(clean)
        return mixed, clean


class MyAudioFolder(MyAudioFolder):
    def __init__(self,
                 root: str,
                 loader: Callable[[str], Any],
                 extensions=('.wav', '.flac'),
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, loader, extensions, transform, target_transform)
        self.classes = ['clean', 'mixed']
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}
        self.find_samples()

    def find_samples(self):
        samples = {}
        for cls in self.classes:
            s = []
            for ext in self.extensions:
                s += glob(join(self.root, '{}/*{}'.format(cls, ext)))
            samples[cls] = sorted(s)
        self.samples = samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        mixed = self.loader(self.samples['mixed'][index])
        clean = self.loader(self.samples['clean'][index])
        if self.transform is not None:
            mixed = self.transform(mixed)
        if self.target_transform is not None:
            clean = self.target_transform(clean)
        return mixed, clean


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    # prepare data
    datamodule.prepare_data()

    # set up data
    datamodule.setup()

    # get train, validation, test dataset
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset

    # get the first sample and target in the train dataset
    x, y = train_dataset[0]

    # display the dimension of sample and target
    print('the dimension of sample: {}'.format(x.shape))
    print('the dimension of target: {}'.format(y.shape))
