import streamlit as st
import streamlit.components.v1 as components
import PIL.Image
from io import BytesIO

# Function for Image Recognition
def image_recognition():
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        # Use your classifier function here
        # For demonstration purposes, let's assume the prediction is 'cat'
        label = 'cat' 
        st.write(f"Prediction: {label}")

# Load custom Streamlit component
def tableau_component():
    return components.html(open("tableau_component/tableau.html").read(), height=800, width = 2000)


def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ["Image Recognition", "Tableau Dashboard"])

    if choice == "Image Recognition":
        image_recognition()
    elif choice == "Tableau Dashboard":
        tableau_component()

if __name__ == "__main__":
    main()
