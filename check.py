import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

try:
    # compile=False dene se version conflicts kam ho jate hain
    model = tf.keras.models.load_model('my_model.h5', compile=False)
    print("\n✅ Mubarak Ho! Colab wala model successfully load ho gaya hai.")
    
    # Model ki shakal check karne ke liye
    model.summary() 
except Exception as e:
    print(f"\n❌ Abhi bhi masla hai: {e}")