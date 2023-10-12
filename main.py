"""
This module can be executed with the command "streamlit run main.py" 
and launch the MVP of our product.
"""

import base64
import io

from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd


from app import classifier

# Load the logo image
LOGO_PATH = "app/images/logo.png"
with open(LOGO_PATH, "rb") as image_file:
    logo_bytes = image_file.read()

# Convert the image to a data URI
logo_b64 = base64.b64encode(logo_bytes).decode()
logo_uri = f"data:image/png;base64,{logo_b64}"

header_html = f"""
<style>
    body {{
        margin-top: 100px;  /* Adjusted value */
    }}
</style>
<div style="position: absolute; top: 10px; right: 10px; text-align: right;">
    <img src="{logo_uri}" alt="logo" style="height: 100px;">
    <p style="font-size: 16px; margin-top: 10px;">CleanR | Driving Change, Diminishing Methane</p>  <!-- Adjusted margin-top value -->
</div>
"""

st.markdown(header_html, unsafe_allow_html=True)


def process_image(img):
    """
    Processes the given image by converting it to RGB and resizing it to 32x32 pixels.

    Parameters:
    img (PIL.Image.Image): The image to be processed.

    Returns:
    PIL.Image.Image: The processed image with 3 channels (RGB) and resized to 32x32 pixels.
    """
    img = img.convert("RGB")  # Ensure the image has 3 channels (RGB)
    img = img.resize((32, 32))  # Resize the image to 32x32 pixels
    return img



# Sidebar for navigation
page = st.sidebar.radio(
    "Navigate",
    options=[
        "Home",
        "Methane Report",
        "Visualize My Emissions",
        "End of Leak Certificate",
        "Detect Leaks",
    ],
)

if page == "Home":
    # Create empty space to push the title down
    for _ in range(8):
        st.text("")
    st.title("CleanR: Get control of your Methane emotions & build your CSRD")
    st.write(
        """Our mission is to help companies diminish their Methane emissions 
        by providing a clear method for **Monitoring, Reporting and Verification**."""
    )

    col1, col2 = st.columns(2)  # Create two columns

    with col1:  # First column
        with st.container():
            st.write("### Methane Report")
            st.write(
                "Generate reports on your methane emissions based on the provided data."
            )
            if st.button("📊 Methane Report"):
                st.experimental_rerun()

        with st.container():
            st.write("### Visualize My Emissions")
            st.write("Visualize your emissions data over time to track your progress.")
            if st.button("📈 Visualize My Emissions"):
                st.experimental_rerun()

    with col2:  # Second column
        with st.container():
            st.write("### End of Leak Certificate")
            st.write("Verify the end of a methane leak and generate a certificate.")
            if st.button("📜 End of Leak Certificate"):
                st.experimental_rerun()

        with st.container():
            st.write("### Detect Leaks")
            st.write("Use our AI model to detect methane leaks from images.")
            if st.button("🔍 Detect Leaks"):
                st.experimental_rerun()


elif page == "Methane Report":
    # Create empty space to push the title down
    for _ in range(8):
        st.text("")
    st.title("Methane Emission Report")

    # This string is a tableau dashboard that contains a visualization
    TABLEAU_EMBED_CODE = """
    <div class='tableauPlaceholder' id='viz1697103145933' style='position: relative'><noscript><a href='#'>
    <img alt='1-Leaks report ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Le&#47;Leaksr
    eport&#47;1-Leaksreport&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  
    style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
    <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' 
    value='Leaksreport&#47;1-Leaksreport' /><param name='tabs' value='no' /><param name='toolbar' value='yes' />
    <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Le&#47;Leaksre
    port&#47;1-Leaksreport&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static
    _image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' />
    <param name='display_count' value='yes' /><param name='language' value='en-GB' /><param name='filter' value='publish=yes' /></object></div>
    <script type='text/javascript'>                    var divElement = document.getElementById('viz1697103145933');
    var vizElement = divElement.getElementsByTagName('object')[0];                    
    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
    else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';}
    else { vizElement.style.width='100%';vizElement.style.height='1927px';} 
    var scriptElement = document.createElement('script');                    
    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>    """
    components.html(TABLEAU_EMBED_CODE, height=1000, width=1600)

elif page == "Visualize My Emissions":
    # Create empty space to push the title down
    for _ in range(8):
        st.text("")
    st.title("Visualize My Emissions")

    # This string is a tableau dashboard that contains a visualization
    TABLEAU_EMBED_CODE = """
    <div class='tableauPlaceholder' id='viz1697113359224' style='position: relative'><noscript><a href='#'>
    <img alt='2-Localize your leaks ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Lo&#
    47;Localizeyourleaks-v2&#47;2-Localizeyourleaks&#47;1_rss.png' style='border: none' /></a></noscript>
    <object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
    <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' 
    value='Localizeyourleaks-v2&#47;2-Localizeyourleaks' /><param name='tabs' value='no' /><param name='toolbar' 
    value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Lo&#47;
    Localizeyourleaks-v2&#47;2-Localizeyourleaks&#47;1.png' /> <param name='animate_transition' value='yes' />
    <param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' 
    value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /><param name='filter' 
    value='publish=yes' /></object></div>                <script type='text/javascript'>  
    var divElement = document.getElementById('viz1697113359224'); var vizElement = divElement.getElementsByTagName('object')[0];  
    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';}
    else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
    else { vizElement.style.width='100%';vizElement.style.height='1727px';}  
    var scriptElement = document.createElement('script');   scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js'; 
     vizElement.parentNode.insertBefore(scriptElement, vizElement);   </script> """
    components.html(TABLEAU_EMBED_CODE, height=1000, width=1600)

elif page == "End of Leak Certificate":
    # Create empty space to push the title down
    for _ in range(8):
        st.text("")
    st.title("End of Leak Certificate")

    col1, col2 = st.columns(2)  # Create two columns
    combined_df = pd.read_csv("app/leak_certificate.csv")
    # Create a dropdown menu for selecting an id_coord
    selected_id = st.selectbox("Select an ID", options=combined_df["id_coord"].unique())

    # Filter the DataFrame based on the selected id_coord
    selected_row = combined_df[combined_df["id_coord"] == selected_id].iloc[0]

    # Obtain the file paths for the images
    path1, path2 = "app/" + selected_row["path1"], "app/" + selected_row["path2"]

    # Display the images in the columns
    with col1:
        with open(path1, "rb") as f:
            tiff_img = Image.open(f)
            buf = io.BytesIO()
            tiff_img.convert("RGB").save(buf, format="PNG")
            st.image(
                buf.getvalue(),
                caption=f"Image 1 from {selected_id}",
                use_column_width=True,
            )

    with col2:
        with open(path2, "rb") as f:
            tiff_img = Image.open(f)
            buf = io.BytesIO()
            tiff_img.convert("RGB").save(buf, format="PNG")
            st.image(
                buf.getvalue(),
                caption=f"Image 2 from {selected_id}",
                use_column_width=True,
            )
        # Obtain the values of plume1 and plume2
    plume1, plume2 = selected_row["plume1"], selected_row["plume2"]

    # Check the conditions and display the appropriate message
    if plume1 == "yes" and plume2 == "yes":
        st.markdown(
            "<div style='color: red;'>"
            "Sorry, as of our last updates, there is still a leak in this position. "
            "Resolve the problem to get a certificate."
            "</div>",
            unsafe_allow_html=True,
        )
    elif plume1 == "yes" and plume2 == "no":
        st.markdown(
            "<div style='color: green;'>"
            "Congratulations, you resolved the leak, you can get your certificate!"
            "</div>",
            unsafe_allow_html=True,
        )
        # Text input fields with default values
        company_name = st.text_input(
            "Name of Company:", value="International Gas Company (IGC)"
        )
        date = st.text_input("Date:", value="13/10/2023")

        # Button to download the PDF
        if st.button("Generate Certificate"):
            CERTIFICATE_PATH = "app/images/certif.pdf"
            with open(CERTIFICATE_PATH, "rb") as f:
                certificate_bytes = f.read()
            st.download_button(
                label="Download Certificate",
                data=certificate_bytes,
                file_name="certificate.pdf",
                mime="application/pdf",
            )


elif page == "Detect Leaks":
    model = classifier.Classifier("app/resnet18.pth")

    # Create empty space to push the title down
    for _ in range(8):
        st.text("")
    st.title("Methane Emission Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "tif"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Predict
        prediction = model.classify_image(uploaded_file)

        # Assume that class 1 corresponds to Methane Emission presence
        if prediction[0] == 1:
            st.write(f"Methane Emission Detected with probability {prediction[1]}")
        else:
            st.write(f"Methane not Emission Detected with probablity {prediction[1]}")
