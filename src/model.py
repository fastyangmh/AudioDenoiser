#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.model import BaseModel, load_from_checkpoint
from os.path import isfile
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchsummary import summary
import torch


#def
def create_model(project_parameters):
    model = UnsupervisedModel(
        optimizers_config=project_parameters.optimizers_config,
        lr=project_parameters.lr,
        lr_schedulers_config=project_parameters.lr_schedulers_config,
        model_name=project_parameters.model_name,
        in_chans=project_parameters.in_chans,
        hidden_chans=project_parameters.hidden_chans,
        chans_scale=project_parameters.chans_scale,
        depth=project_parameters.depth,
        loss_function_name=project_parameters.loss_function_name,
        sample_rate=project_parameters.sample_rate)
    if project_parameters.checkpoint_path is not None:
        if isfile(project_parameters.checkpoint_path):
            model = load_from_checkpoint(
                device=project_parameters.device,
                checkpoint_path=project_parameters.checkpoint_path,
                model=model)
        else:
            assert False, 'please check the checkpoint_path argument.\nthe checkpoint_path value is {}.'.format(
                project_parameters.checkpoint_path)
    return model


#class
class UnsupervisedModel(BaseModel):
    def __init__(self, optimizers_config, lr, lr_schedulers_config, model_name,
                 in_chans, hidden_chans, chans_scale, depth,
                 loss_function_name, sample_rate) -> None:
        super().__init__(optimizers_config=optimizers_config,
                         lr=lr,
                         lr_schedulers_config=lr_schedulers_config)
        self.backbone_model = self.create_backbone_model(
            model_name=model_name,
            in_chans=in_chans,
            hidden_chans=hidden_chans,
            chans_scale=chans_scale,
            depth=depth)
        self.loss_function = self.create_loss_function(
            loss_function_name=loss_function_name)
        self.pesq_function = PerceptualEvaluationSpeechQuality(fs=sample_rate,
                                                               mode='wb')

    def create_backbone_model(self, model_name, in_chans, hidden_chans,
                              chans_scale, depth):
        if isfile(model_name):
            class_name = self.import_class_from_file(filepath=model_name)
            backbone_model = class_name(in_chans=in_chans,
                                        hidden_chans=hidden_chans,
                                        chans_scale=chans_scale,
                                        depth=depth)
        else:
            assert False, 'please check the model_name argument.\nthe model_name value is {}.'.format(
                model_name)
        return backbone_model

    def forward(self, x):
        return self.backbone_model(x)

    def shared_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        pesq = self.pesq_function(y_hat, y)
        return loss, pesq

    def training_step(self, batch, batch_idx):
        loss, pesq = self.shared_step(batch=batch)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('train_pesq',
                 pesq,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pesq = self.shared_step(batch=batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_pesq', pesq, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, pesq = self.shared_step(batch=batch)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_pesq', pesq, prog_bar=True)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    summary(model=model,
            input_size=(project_parameters.in_chans,
                        project_parameters.sample_rate),
            device='cpu')

    # create input data
    x = torch.rand(project_parameters.batch_size, project_parameters.in_chans,
                   project_parameters.sample_rate)

    # get model output
    y_hat = model(x)

    # display the dimension of input and output
    print('the dimension of input: {}'.format(x.shape))
    print('the dimension of output: {}'.format(y_hat.shape))
