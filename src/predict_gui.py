#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.predict_gui import BasePredictGUI
from src.predict import Predict
from DeepLearningTemplate.data_preparation import AudioLoader, parse_transforms
from tkinter import Button, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from playsound import playsound
import tkinter as tk
import sounddevice as sd
import gradio as gr


# class
class PredictGUI(BasePredictGUI):
    def __init__(self, project_parameters) -> None:
        super().__init__(extensions=('.wav', '.flac'))
        self.predictor = Predict(project_parameters=project_parameters)
        self.loader = AudioLoader(sample_rate=project_parameters.sample_rate)
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.sample_rate = project_parameters.sample_rate
        self.web_interface = project_parameters.web_interface
        self.examples = project_parameters.examples if len(
            project_parameters.examples) else None

        # button
        self.play_button = Button(master=self.window,
                                  text='Play',
                                  command=self.play)
        self.denoise_button = self.recognize_button
        self.denoise_button.config(text='Denoise', command=self.denoise)

        # matplotlib canvas
        # this is Tkinter default background-color
        facecolor = (0.9254760742, 0.9254760742, 0.9254760742)
        figsize = np.array([12, 4]) * project_parameters.in_chans
        self.image_canvas = FigureCanvasTkAgg(Figure(figsize=figsize,
                                                     facecolor=facecolor),
                                              master=self.window)

    def display(self):
        waveform = self.loader(path=self.filepath)
        rows, cols = waveform.shape[0], 1
        for idx in range(1, rows * cols + 1):
            subplot = self.image_canvas.figure.add_subplot(rows, cols, idx)
            # plot waveform
            subplot.title.set_text(
                'channel {} waveform'.format((idx - 1) // cols + 1))
            subplot.set_xlabel('time')
            subplot.set_ylabel('amplitude')
            time = np.linspace(
                0, len(waveform[(idx - 1) // cols]),
                len(waveform[(idx - 1) // cols])) / self.sample_rate
            subplot.plot(time, waveform[(idx - 1) // cols])
        self.image_canvas.draw()
        self.waveform = waveform

    def open_file(self):
        super().open_file()
        self.display()

    def reset_widget(self):
        super().reset_widget()
        self.image_canvas.figure.clear()

    def display_output(self, sample_hat):
        rows, cols = self.waveform.shape[0], 2
        for idx in range(1, rows * cols + 1):
            subplot = self.image_canvas.figure.add_subplot(rows, cols, idx)
            if idx % cols == 1:
                # plot original waveform
                subplot.title.set_text(
                    'channel {} original waveform'.format((idx - 1) // cols +
                                                          1))
                subplot.set_xlabel('time')
                subplot.set_ylabel('amplitude')
                time = np.linspace(
                    0, len(self.waveform[(idx - 1) // cols]),
                    len(self.waveform[(idx - 1) // cols])) / self.sample_rate
                subplot.plot(time, self.waveform[(idx - 1) // cols])
            if idx % cols == 0:
                # plot predicted waveform
                subplot.title.set_text(
                    'channel {} denoised waveform'.format((idx - 1) // cols +
                                                          1))
                subplot.set_xlabel('time')
                subplot.set_ylabel('amplitude')
                time = np.linspace(
                    0, len(sample_hat[(idx - 1) // cols]),
                    len(sample_hat[(idx - 1) // cols])) / self.sample_rate
                subplot.plot(time, sample_hat[(idx - 1) // cols])
        self.image_canvas.draw()

    def denoise(self):
        if self.filepath is not None:
            #sample_hat dimension is (in_chans, length)
            sample_hat = self.predictor.predict(inputs=self.filepath)
            self.reset_widget()
            self.display_output(sample_hat=sample_hat)
            sd.play(data=sample_hat.mean(0), samplerate=self.sample_rate)
        else:
            messagebox.showerror(title='Error!', message='please open a file!')

    def play(self):
        if self.filepath is not None:
            playsound(sound=self.filepath, block=True)
        else:
            messagebox.showerror(title='Error!', message='please open a file!')

    def inference(self, inputs):
        #sample_hat dimension is (in_chans, length)
        sample_hat = self.predictor.predict(inputs=inputs)
        sample_hat = sample_hat.cpu().data.numpy()
        # gradio API only support 2 channels audio
        in_chans, _ = sample_hat.shape
        if in_chans != 2:
            sample_hat = sample_hat.mean(0)
            sample_hat = np.concatenate([sample_hat[None] for idx in range(2)],
                                        0)
        #convert the data type of sample_hat for gradio API
        sample_hat = (sample_hat * np.iinfo(np.int16).max).astype(np.int16)
        #convert the dimension of sample_hat from (in_chans, length) to (length, in_chans)
        sample_hat = sample_hat.T
        return (self.sample_rate, sample_hat)

    def run(self):
        if self.web_interface:
            gr.Interface(fn=self.inference,
                         inputs=gr.inputs.Audio(source='microphone',
                                                type='filepath'),
                         outputs=gr.outputs.Audio(type='numpy'),
                         examples=self.examples,
                         interpretation="default").launch(share=True,
                                                          inbrowser=True)
        else:
            # NW
            self.open_file_button.pack(anchor=tk.NW)
            self.denoise_button.pack(anchor=tk.NW)
            self.play_button.pack(anchor=tk.NW)

            # N
            self.filepath_label.pack(anchor=tk.N)
            self.image_canvas.get_tk_widget().pack(anchor=tk.N)
            self.predicted_label.pack(anchor=tk.N)
            self.result_label.pack(anchor=tk.N)

            # run
            super().run()


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # launch prediction gui
    PredictGUI(project_parameters=project_parameters).run()
