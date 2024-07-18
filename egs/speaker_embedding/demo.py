import argparse
import time

import gradio as gr
import numpy as np
import onnxruntime
import torch

from puresound.audio.io import AudioIO


def cos_similarity(s1: torch.Tensor, s2: torch.Tensor):
    return (s1 * s2).sum(axis=-1) / np.sqrt(
        (s1 * s1).sum(axis=-1) * (s2 * s2).sum(axis=-1)
    )


def inference(onnx_sess, anchor_wav, test_wav, sr=16000, norm_gain=-22, threshold=0.8):
    embedding_list = []

    for wav in [anchor_wav, test_wav]:
        localtime = time.strftime("%Y-%m-%d-%I-%M-%S", time.localtime())
        print(f"Uploaded audio: {wav}, {localtime}")
        wav, wav_sr = AudioIO.open(f_path=wav, target_lvl=norm_gain, resample_to=sr)

        if wav.shape[0] != 0:
            wav = wav[0].view(1, -1)

        wav = wav.numpy()
        input_name = onnx_sess.get_inputs()[0].name
        ort_inputs = {input_name: wav}
        ort_outs = onnx_sess.run(None, ort_inputs)
        embedding_list.append(ort_outs[0])

    scores = cos_similarity(s1=embedding_list[0], s2=embedding_list[1])
    scores = round(scores.item(), 3)
    if scores > threshold:
        return f"PASS, Cosine similarity is: {scores}"
    else:
        return f"Fail, Cosine similarity is: {scores}"


def main(args):
    onnx_sess = onnxruntime.InferenceSession(args.onnx_path)

    warp_inference = lambda enroll_wav, test_wav, norm_gain, threshold: inference(
        onnx_sess=onnx_sess,
        anchor_wav=enroll_wav,
        test_wav=test_wav,
        norm_gain=norm_gain,
        threshold=threshold,
    )

    gr.Interface(
        fn=warp_inference,
        title="PureSound's Speaker Verification",
        description="Note: upload audio must be 16k 16-int format only.",
        inputs=[
            gr.Audio(
                label="Upload Enrollment Speech",
                type="filepath",
                show_download_button=True,
            ),
            gr.Audio(
                label="Upload Testing Speech",
                type="filepath",
                show_download_button=True,
            ),
            gr.Slider(
                label="Nomalized gain to (dB)",
                minimum=-40,
                maximum=-16,
                value=-22,
                step=2,
            ),
            gr.Slider(
                label="Cosine Threshold",
                minimum=0.1,
                maximum=0.95,
                value=0.8,
                step=0.05,
            ),
        ],
        outputs=[
            gr.Textbox(label="Scores"),
        ],
    ).launch(server_name=args.address, server_port=args.port, share=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", type=str)
    parser.add_argument("--address", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6666)
    args = parser.parse_args()
    main(args)
