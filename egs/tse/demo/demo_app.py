import sys
import threading
import time
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
import sv_ttk
import torch
import torchaudio
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from playsound import playsound
from torchaudio.io import StreamReader

sys.path.insert(0, '..')
from model import init_model


class DemoAPP():
    def __init__(self):
        self.window = tk.Tk()
        sv_ttk.set_theme("light")  # Set light them
        sv_ttk.use_light_theme()  # Set light theme
        # sv_ttk.toggle_theme()  # Toggle between dark and light theme
        self.window.title("Recording tool version 0.1")
        self.window.resizable(width=False, height=True)

        self.all_block_dct = {}
        self.init_app()
        self.window.mainloop()

    def init_app(self):
        self.task_model = init_model('tse_skim_v0_causal')
        ckpt = torch.load('skim_causal_460.ckpt', map_location='cpu')
        self.task_model.load_state_dict(ckpt['state_dict'], strict=False)
        self.task_model.eval()

        self.continue_recording = False
        self.speaker_embedding = None
        self.enroll_wav = None
        self.top_screen()
        self.enroll_block()

    def top_screen(self):
        global screen_img
        _screen_img = Image.open('screen.png')
        screen_img = ImageTk.PhotoImage(_screen_img.resize((550, 150)))
        label_screen_img = tk.Label(self.window, image=screen_img)
        label_screen_img.grid(row=0, column=0)

    def enroll_block(self,):
        title_txt_frame = tk.Label(self.window)
        title_txt_frame.grid(row=1, column=0, padx=5, pady=5)
        
        global sucess_img
        if self.speaker_embedding is None:
            enroll_done_img = Image.open('fail.png')
        else:
            enroll_done_img = Image.open('success.png')

        sucess_img = ImageTk.PhotoImage(enroll_done_img.resize((65, 65)))
        label_sucess_img = tk.Label(title_txt_frame, image=sucess_img)
        label_sucess_img.grid(row=0, column=0, rowspan=2, ipadx=10)

        def _record_button_func():
            if not self.continue_recording:
                """Click to record enroll speech."""
                self.continue_recording = True
                threading._start_new_thread(self._record_enroll, ('./enroll.wav',)) # open a thread to record
                time.sleep(0.5)
                record_button['text'] = 'Stop!' # change button icon
                record_button['fg'] = 'red'

            else:
                """Click to stop record."""
                self.continue_recording = False

                # extract speaker embedding
                time.sleep(0.5)
                with torch.no_grad():
                    self.speaker_embedding = self.task_model.inference_tse_embedding(self.enroll_wav).squeeze(-1)

                np.savetxt('./speaker.txt', self.speaker_embedding.numpy())

                record_button['text'] = 'Record'
                record_button['fg'] = 'black'

                # Update enroll pic
                if self.speaker_embedding is None:
                    enroll_done_img = Image.open('fail.png')
                else:
                    enroll_done_img = Image.open('success.png')
                
                global sucess_img
                sucess_img = ImageTk.PhotoImage(enroll_done_img.resize((65, 65)))
                label_sucess_img['image'] = sucess_img

        def _clear_button_func():
            del self.enroll_wav
            del self.speaker_embedding
            
            enroll_done_img = Image.open('fail.png')
            global sucess_img
            sucess_img = ImageTk.PhotoImage(enroll_done_img.resize((65, 65)))
            label_sucess_img['image'] = sucess_img
        
        def _show_button_func():
            plt.specgram(self.enroll_wav.squeeze(), NFFT=512, Fs=16000, cmap='jet')
            plt.title('Spectrogram of enrollment speech')
            plt.xlabel('time (seconds)')
            plt.ylabel('frequency (Hz)')
            threading._start_new_thread(playsound, ('./enroll.wav',))
            plt.show()
        
        def _onoff_button_func():
            global onoff_img

            if not self.continue_recording:
                """Click to record enroll speech."""
                self.continue_recording = True
                threading._start_new_thread(self._record, ('./noisy.wav',)) # open a thread to record
                time.sleep(0.5)
                onoff_button['text'] = 'On!' # change button icon
                onoff_button['fg'] = 'green'

            else:
                """Click to stop record."""
                self.continue_recording = False

                # extract speaker embedding
                time.sleep(0.5)
                with torch.no_grad():
                    self.enh_wav = self.task_model.inference(self.noisy_wav, self.enroll_wav)
                    torchaudio.save('./enh.wav', self.enh_wav.view(1, -1), 16000)
                
                plt.subplot(2,1,1)
                plt.specgram(self.noisy_wav.squeeze(), NFFT=512, Fs=16000, cmap='jet')
                plt.title('Spectrogram of noisy speech')
                plt.xlabel('time (seconds)')
                plt.ylabel('frequency (Hz)')
                plt.subplot(2,1,2)
                plt.specgram(self.enh_wav.squeeze(), NFFT=512, Fs=16000, cmap='jet')
                plt.title('Spectrogram of target speech')
                plt.xlabel('time (seconds)')
                plt.ylabel('frequency (Hz)')
                threading._start_new_thread(playsound, ('./enh.wav',))
                plt.show()
                onoff_button['text'] = 'Off'
                onoff_button['fg'] = 'red'

        record_button = tk.Button(title_txt_frame, width='6', text='Record', font=("Arial", 16, 'bold'), command=_record_button_func)
        record_button.grid(row=1, column=1, pady=5, ipadx=10)
        
        clear_button = tk.Button(title_txt_frame, width='6', text='Clear', font=("Arial", 16, 'bold'), command=_clear_button_func)
        clear_button.grid(row=1, column=2, pady=5, ipadx=10)
        
        show_button = tk.Button(title_txt_frame, width='6', text='Show', font=("Arial", 16, 'bold'), command=_show_button_func)
        show_button.grid(row=1, column=3, pady=5, ipadx=10)

        onoff_button = tk.Button(title_txt_frame, width='6', text='Off', fg='red', font=("Arial", 16, 'bold'), command=_onoff_button_func)
        onoff_button.grid(row=1, column=4, pady=5, columnspan=3, ipadx=10)

        global enroll_text
        enroll_text = f"Target speech extraction is a solution for helping people sound pure and clean, even though you are in a noisy environment or cocktail party."
        context_txt = tk.Label(title_txt_frame, text=enroll_text, justify='left', anchor="nw",  wraplength=450, font=("Arial", 18))
        context_txt.grid(row=0, column=1, columnspan=4, ipadx=10)
    
    def _record_enroll(self, save_path, format='avfoundation', src=':1', segment_length=512, sample_rate=16000):
        wav = []

        print("Building StreamReader...")
        streamer = StreamReader(src, format=format)
        streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=sample_rate)

        print("Streaming...")
        print()

        stream_iterator = streamer.stream(timeout=-1, backoff=1.0)
        while self.continue_recording:
            (chunk,) = next(stream_iterator)
            wav.append(chunk)
        
        wav = torch.cat(wav).view(1, -1)
        torchaudio.save(save_path, wav, sample_rate)
        print('Streaming closed')

        self.enroll_wav = wav
        
    def _record(self, save_path, format='avfoundation', src=':1', segment_length=512, sample_rate=16000):
        global stream_wav
        stream_wav = []

        print("Building StreamReader...")
        streamer = StreamReader(src, format=format)
        streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=sample_rate)

        # print(streamer.get_src_stream_info(0))
        # print(streamer.get_out_stream_info(0))

        print("Streaming...")
        print()

        stream_iterator = streamer.stream(timeout=-1, backoff=1.0)
        while self.continue_recording:
            (chunk,) = next(stream_iterator)
            stream_wav.append(chunk)

        wav = torch.cat(stream_wav).view(1, -1)
        torchaudio.save(save_path, wav, sample_rate)
        print('Streaming closed')

        self.noisy_wav = wav

if __name__ == '__main__':
    DemoAPP()
