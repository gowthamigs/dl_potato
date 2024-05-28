import streamlit as st
import tensorflow as tf
import numpy as np


# TensorFlow Model Prediction
def model_prediction(test_image):
  """Predicts the class of a plant disease image using a loaded TensorFlow model.

  Args:
      test_image: Path to the image file for prediction.

  Returns:
      Integer: Index of the predicted class from the loaded model.
  """
  try:
    model = tf.keras.models.load_model("potato.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element
  except Exception as e:
    st.error(f"Error: {e}")
    return None  # Indicate error during prediction

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
  st.header("PLANT DISEASE RECOGNITION SYSTEM")
  image_path = "home_page.jpeg"  # Ensure this image exists
  st.image(image_path, use_column_width=True)
  st.markdown("""
  Welcome to the Plant Disease Recognition System! 

  Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

  ### How It Works
  1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
  2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
  3. **Results:** View the results and recommendations for further action.

  ### Why Choose Us?
  - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
  - **User-Friendly:** Simple and intuitive interface for seamless user experience.
  - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

  ### Get Started
  Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

  ### About Us
  Learn more about the project, our team, and our goals on the **About** page.
  """)

# About Project
elif app_mode == "About":
  st.header("About")
  st.markdown("""
  #### About Dataset
  This dataset X (replace with your dataset description)
  """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
  st.header("Disease Recognition")
  test_image = st.file_uploader("Choose an Image:", type=['jpg', 'png'])
  if test_image is not None:
    st.image(test_image, width=4, use_column_width=True)
    if st.button("Predict"):
      with st.spinner("Predicting..."):
        result_index = model_prediction(test_image)
      if result_index is not None:
        # Reading Labels (Replace with your actual class names)
       # class_name = ['Apple___Apple_scab', 'Apple___Black_rot', '...']  # Add your class names here
        class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
      else:
        st.error("An error occurred during prediction.")

