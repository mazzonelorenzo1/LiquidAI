# Install the proper packages before running the code, you can find them in the Requirements.py file
# I suggest to run this code on Google Colab with the GPU runtime enabled to avoid problems with CUDA

import torch
import torchaudio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

# Load models
HF_REPO = "LiquidAI/LFM2-Audio-1.5B"

processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

# Set up inputs for the model
chat = ChatState(processor)

chat.new_turn("system")
chat.add_text("Respond_with_interleaved_text_and_audio.")
chat.end_turn()

chat.new_turn("user")

# Fix the demo file with the proper directory path
wav, sampling_rate = torchaudio.load("/content/Liquid_question.m4a")

# Convert stereo to mono if necessary
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
chat.add_audio(wav, sampling_rate)
chat.end_turn()

chat.new_turn("assistant")

# Generate text and audio tokens.
text_out: list[torch.Tensor] = []
audio_out: list[torch.Tensor] = []
modality_out: list[LFMModality] = []
for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)
        text_out.append(t)
        modality_out.append(LFMModality.TEXT)
    else:
        audio_out.append(t)
        modality_out.append(LFMModality.AUDIO_OUT)

# Detokenize audio, removing the last "end-of-audio" codes
# Mimi returns audio at 24kHz
mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
with torch.no_grad():
    waveform = processor.mimi.decode(mimi_codes)[0]
torchaudio.save("answer_Liquid.wav", waveform.cpu(), 24_000)

# Append newly generated tokens to chat history
chat.append(
    text = torch.stack(text_out, 1),
    audio_out = torch.stack(audio_out, 1),
    modality_flag = torch.tensor(modality_out),
)
chat.end_turn()

# Start second turn
chat.new_turn("user")

# Fix the demo file with the proper directory path
wav, sampling_rate = torchaudio.load("/content/GPU_question.m4a")
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
chat.add_audio(wav, sampling_rate)
chat.end_turn()

chat.new_turn("assistant")

# Generate second turn text and audio tokens.
audio_out: list[torch.Tensor] = []
for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)
    else:
        audio_out.append(t)

# Detokenize second turn audio, removing the last "end-of-audio" codes
mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
with torch.no_grad():
    waveform = processor.mimi.decode(mimi_codes)[0]
torchaudio.save("answer_GPU.wav", waveform.cpu(), 24_000)

# Start third turn
chat.new_turn("user")

# Fix the demo file with the proper directory path
wav, sampling_rate = torchaudio.load("/content/Cuda_error.m4a")
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
chat.add_audio(wav, sampling_rate)
chat.end_turn()

chat.new_turn("assistant")

# Generate third turn text and audio tokens.
audio_out: list[torch.Tensor] = []
for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)
    else:
        audio_out.append(t)

# Detokenize third audio, removing the last "end-of-audio" codes
mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
with torch.no_grad():
    waveform = processor.mimi.decode(mimi_codes)[0]
torchaudio.save("answer_Cuda.wav", waveform.cpu(), 24_000)

# Start fourth turn
chat.new_turn("user")

# I tried text input to see if that works too, change it as you wish
chat.add_text("Okay_I_will_try_with_this_solution_thanks")
chat.end_turn()

chat.new_turn("assistant")

# Generate fourth turn text and audio tokens.
audio_out: list[torch.Tensor] = []
for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)
    else:
        audio_out.append(t)

# Detokenize fourth turn audio, removing the last "end-of-audio" codes
mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
with torch.no_grad():
    waveform = processor.mimi.decode(mimi_codes)[0]
torchaudio.save("answer_solution.wav", waveform.cpu(), 24_000)
