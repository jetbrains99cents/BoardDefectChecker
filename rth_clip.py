# rth_clip.py
import sys
import os

# When running from a PyInstaller bundle, files are extracted into sys._MEIPASS.
if hasattr(sys, '_MEIPASS'):
    vocab_path = os.path.join(sys._MEIPASS, 'clip', 'bpe_simple_vocab_16e6.txt.gz')
    try:
        # Import the tokenizer module and update the path it uses.
        import clip.simple_tokenizer as st
        st.VOCAB_FILE = vocab_path
    except Exception as e:
        print("Could not set VOCAB_FILE in clip.simple_tokenizer:", e)
