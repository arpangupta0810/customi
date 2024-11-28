import streamlit as st
import os
import base64
import torch
import cv2
import numpy as np
from diffusers import DiffusionPipeline
from groq import Groq

# Set the API key for Groq
os.environ["GROQ_API_KEY"] = "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"  # Replace with your Groq API Key

# Function to get the Groq client
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key)

# Streamlit page configuration
st.title("Image Description Refinement and Generation")
st.write("Upload an image, refine the description, and generate a new image.")

# Step 1: Image upload by user
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Step 2: Show the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image temporarily
    image_path = uploaded_file.name
    cv2.imwrite(image_path, image)

    # Step 3: Encode the image to base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    # Step 4: Get image description from Groq
    client = get_groq_client()
    completion = client.chat.completions.create(
        model="llava-v1.5-7b-4096-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    # Display the generated description
    initial_description = completion.choices[0].message.content
    st.write("### Initial Description:")
    st.write(initial_description)

    # Step 5: User input for refinement
    refinement_prompt = st.text_input("Enter a prompt to refine the description:", "")
    if refinement_prompt:
        # Step 6: Refining the description using the Llama model
        lama_completion = client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"Refine the following description based on the prompt:\n\nDescription: {initial_description}\n\nPrompt: {refinement_prompt}"
                }
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        refined_description = lama_completion.choices[0].message.content
        st.write("### Refined Description:")
        st.write(refined_description)

        # Step 7: Use refined description to generate image
        st.write("Generating an image based on the refined description...")
        if torch.cuda.is_available():
            pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
        else:
            pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")  # For CPU usage

        # Generate the image
        generated_image = pipe(refined_description).images[0]
        image_path = "generated_image.png"
        generated_image.save(image_path)

        # Convert the generated image to OpenCV format for display
        generated_image_cv2 = cv2.imread(image_path)
        st.image(generated_image_cv2, channels="BGR", caption="Generated Image", use_column_width=True)
        st.write(f"Image saved as {image_path}")
