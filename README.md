# Indic Parler-TTS (CPU Setup Guide)

This guide explains how to install and use the ai4bharat/indic-parler-tts model on a CPU-only Linux machine and generate playable audio output.

------------------------------------------------------------
STEP 1 — Create Project Folder
------------------------------------------------------------

mkdir indic_tts_test
cd indic_tts_test

------------------------------------------------------------
STEP 2 — Create Virtual Environment
------------------------------------------------------------

python3 -m venv localenv
source localenv/bin/activate

You should now see:
(localenv)

------------------------------------------------------------
STEP 3 — Install Required Packages
------------------------------------------------------------

Upgrade pip:

pip install --upgrade pip

Install PyTorch (CPU version):

pip install torch torchvision torchaudio

Install required libraries:

pip install transformers soundfile huggingface_hub

Install Parler-TTS:

pip install git+https://github.com/huggingface/parler-tts.git

------------------------------------------------------------
STEP 4 — Request Model Access
------------------------------------------------------------

Open in browser:

https://huggingface.co/ai4bharat/indic-parler-tts

Click:
Request Access

Wait for approval.

------------------------------------------------------------
STEP 5 — Create Hugging Face Token
------------------------------------------------------------

Go to:

https://huggingface.co/settings/tokens

Create:
New Token
Type: Read

Copy the token (starts with hf_...)

------------------------------------------------------------
STEP 6 — Login in Terminal
------------------------------------------------------------

huggingface-cli login

Paste your token.

Verify login:

huggingface-cli whoami

If it prints your username, authentication is successful.

------------------------------------------------------------
STEP 7 — Create Python Script
------------------------------------------------------------

Create file:

nano cpu_test_tts.py

Paste the following code:

------------------------------------------------------------

import torch
import numpy as np
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# CPU settings
device = "cpu"
torch.set_num_threads(4)
print("Running on CPU")

# Load model
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts",
    torch_dtype=torch.float32
).to(device)

model.eval()

# Load tokenizers
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

# Kannada prompt
prompt = """ನಮಸ್ಕಾರ, ನಾನು Vishwas Kumar, The Hiring Companyದಿಂದ ಕರೆ ಮಾಡುತ್ತಿದ್ದೇನೆ.
Everest Fleet ಮೂಲಕ Uber driver hiring ನಡೆಯುತ್ತಿದೆ Bengaluru ಮತ್ತು Delhiನಲ್ಲಿ.
Driver job opportunity ಬಗ್ಗೆ ನಿಮಗೆ ಆಸಕ್ತಿ ಇದೆಯಾ?"""

# Female speaker, polite, conversational
description = """
Anu speaks in a polite and friendly conversational tone
with a natural Kannada accent and very clear audio.
"""

# Tokenize
description_inputs = description_tokenizer(description, return_tensors="pt")
prompt_inputs = tokenizer(prompt, return_tensors="pt")

# Generate
with torch.no_grad():
    generation = model.generate(
        input_ids=description_inputs.input_ids,
        attention_mask=description_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask,
        do_sample=True,
        temperature=1.0,
        top_p=0.9
    )

# Convert to waveform
audio = generation.cpu().numpy().squeeze()

# Normalize audio
max_val = np.max(np.abs(audio))
if max_val > 0:
    audio = audio / max_val

# Save as 16-bit PCM WAV
sf.write(
    "cpu_output.wav",
    audio,
    model.config.sampling_rate,
    subtype="PCM_16"
)

print("Audio saved as cpu_output.wav")

------------------------------------------------------------

Save and exit:
CTRL + X
Y
ENTER

------------------------------------------------------------
STEP 8 — Run Script
------------------------------------------------------------

python cpu_test_tts.py

First run will:
- Download ~3.75GB model
- Take 1–2 minutes on CPU
- Generate audio file

------------------------------------------------------------
STEP 9 — Install Audio Player (If Needed)
------------------------------------------------------------

sudo apt install ffmpeg

------------------------------------------------------------
STEP 10 — Play Audio
------------------------------------------------------------

ffplay cpu_output.wav

OR

vlc cpu_output.wav

------------------------------------------------------------
SYSTEM REQUIREMENTS
------------------------------------------------------------

Minimum:
- 8GB RAM
- 5GB free disk space
- 4+ CPU cores recommended

------------------------------------------------------------
HOW IT WORKS
------------------------------------------------------------

Transcript + Style Description
        ↓
Flan-T5 Text Encoder
        ↓
Parler TTS Decoder
        ↓
DAC Audio Codec (44.1kHz)
        ↓
cpu_output.wav
        ↓
Audio Playback

------------------------------------------------------------
OUTPUT
------------------------------------------------------------

The system generates a polite female Kannada voice (Anu) speaking the provided hiring script in a natural conversational tone.

------------------------------------------------------------
END
------------------------------------------------------------

