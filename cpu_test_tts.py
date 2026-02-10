# import torch
# import numpy as np
# import soundfile as sf
# from parler_tts import ParlerTTSForConditionalGeneration
# from transformers import AutoTokenizer

# # ==============================
# # Force CPU
# # ==============================
# device = "cpu"
# print("Running on CPU")

# # Optional: limit CPU threads (adjust if needed)
# torch.set_num_threads(4)

# # ==============================
# # Load Model
# # ==============================
# model = ParlerTTSForConditionalGeneration.from_pretrained(
#     "ai4bharat/indic-parler-tts",
#     torch_dtype=torch.float32
# ).to(device)    

# model.eval()

# # ==============================
# # Load Tokenizers
# # ==============================
# tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
# description_tokenizer = AutoTokenizer.from_pretrained(
#     model.config.text_encoder._name_or_path
# )

# # ==============================
# # Kannada Prompt (Your Script)
# # ==============================
# prompt = """ನಮಸ್ಕಾರ, ನಾನು Vishwas Kumar, The Hiring Companyದಿಂದ ಕರೆ ಮಾಡುತ್ತಿದ್ದೇನೆ.
# Everest Fleet ಮೂಲಕ Uber driver hiring ನಡೆಯುತ್ತಿದೆ Bengaluru ಮತ್ತು Delhiನಲ್ಲಿ.
# Driver job opportunity ಬಗ್ಗೆ ನಿಮಗೆ ಆಸಕ್ತಿ ಇದೆಯಾ?"""

# description = "Suresh speaks in a natural Kannada tone with moderate speed and very clear audio."

# # ==============================
# # Tokenize Inputs
# # ==============================
# description_inputs = description_tokenizer(description, return_tensors="pt")
# prompt_inputs = tokenizer(prompt, return_tensors="pt")

# # ==============================
# # Generate Audio
# # ==============================
# with torch.no_grad():
#     generation = model.generate(
#         input_ids=description_inputs.input_ids,
#         attention_mask=description_inputs.attention_mask,
#         prompt_input_ids=prompt_inputs.input_ids,
#         prompt_attention_mask=prompt_inputs.attention_mask,
#         do_sample=True,
#         temperature=1.0
#     )

# # ==============================
# # Convert to waveform
# # ==============================
# audio = generation.cpu().numpy().squeeze()

# print("Audio shape:", audio.shape)
# print("Min value:", audio.min())
# print("Max value:", audio.max())
# print("Mean value:", audio.mean())

# # ==============================
# # Fix silent/low-volume issue
# # ==============================
# max_val = np.max(np.abs(audio))

# if max_val > 0:
#     audio = audio / max_val
# else:
#     print("Warning: Audio is silent (all zeros)")

# # ==============================
# # Save file
# # ==============================
# sf.write("cpu_output.wav", audio, model.config.sampling_rate, subtype="PCM_16")


# print("Sampling rate:", model.config.sampling_rate)
# print("Audio saved as cpu_output.wav")


import torch
import numpy as np
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# ======================================
# CPU SETTINGS
# ======================================
device = "cpu"
print("Running on CPU")
torch.set_num_threads(4)

# ======================================
# LOAD MODEL
# ======================================
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts",
    torch_dtype=torch.float32
).to(device)

model.eval()

# ======================================
# LOAD TOKENIZERS
# ======================================
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

# ======================================
# PROMPT (UNCHANGED AS REQUESTED)
# ======================================
prompt = """ನಮಸ್ಕಾರ, ನಾನು Vishwas Kumar, The Hiring Companyದಿಂದ ಕರೆ ಮಾಡುತ್ತಿದ್ದೇನೆ.
Everest Fleet ಮೂಲಕ Uber driver hiring ನಡೆಯುತ್ತಿದೆ Bengaluru ಮತ್ತು Delhiನಲ್ಲಿ.
Driver job opportunity ಬಗ್ಗೆ ನಿಮಗೆ ಆಸಕ್ತಿ ಇದೆಯಾ?"""

# ======================================
# FEMALE SPEAKER - POLITE - CONVERSATION
# Using recommended Kannada female voice: Anu
# ======================================
description = """
Anu speaks very politely in a calm conversational tone with a soft natural Kannada accent.
Her voice is warm, slightly expressive, moderately paced and recorded with very clear audio.
"""

# ======================================
# TOKENIZATION
# ======================================
description_inputs = description_tokenizer(description, return_tensors="pt")
prompt_inputs = tokenizer(prompt, return_tensors="pt")

# ======================================
# GENERATE AUDIO
# ======================================
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

# ======================================
# CONVERT TO WAVEFORM
# ======================================
audio = generation.cpu().numpy().squeeze()

print("Audio shape:", audio.shape)
print("Min value:", audio.min())
print("Max value:", audio.max())

# ======================================
# NORMALIZE (Prevent low volume)
# ======================================
max_val = np.max(np.abs(audio))
if max_val > 0:
    audio = audio / max_val
else:
    print("Warning: Audio seems silent")

# ======================================
# SAVE AS STANDARD 16-BIT PCM
# ======================================
sf.write(
    "kannada_female.wav",
    audio,
    model.config.sampling_rate,
    subtype="PCM_16"
)

print("Sampling rate:", model.config.sampling_rate)
print("Audio saved as cpu_output.wav")
