"""
Suppress TensorFlow informational messages for cleaner console output.
Run this before launching the Streamlit app: python suppress_tf_warnings.py
Or set these environment variables manually.
"""
import os

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

# Suppress TensorFlow deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

print("✓ TensorFlow warning suppression configured!")
print("\nEnvironment variables set:")
print(f"  TF_CPP_MIN_LOG_LEVEL = {os.environ.get('TF_CPP_MIN_LOG_LEVEL')}")
print(f"  TF_ENABLE_ONEDNN_OPTS = {os.environ.get('TF_ENABLE_ONEDNN_OPTS')}")
print("\nNow run: streamlit run app.py")
