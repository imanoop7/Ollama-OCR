import streamlit as st
from ocr_processor import OCRProcessor
import tempfile
import os
from PIL import Image
import json

# Page configuration
st.set_page_config(
    page_title="OCR with Ollama",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .main {
        background-color: #f8f9fa;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #ccc;
        border-radius: 10px;
        background-color: #ffffff;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        padding: 1rem;
    }
    .gallery-item {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 0.5rem;
        background: white;
    }
    </style>
    """, unsafe_allow_html=True)

def get_available_models():
    return ["llava:7b", "llama3.2-vision:11b"]

def process_single_image(processor, image_path, format_type, enable_preprocessing):
    """Process a single image and return the result"""
    try:
        result = processor.process_image(
            image_path=image_path,
            format_type=format_type,
            preprocess=enable_preprocessing
        )
        return result
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_batch_images(processor, image_paths, format_type, enable_preprocessing):
    """Process multiple images and return results"""
    try:
        results = processor.process_batch(
            input_path=image_paths,
            format_type=format_type,
            preprocess=enable_preprocessing
        )
        return results
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("🔍 Vision OCR Lab")
    st.markdown("<p style='text-align: center; color: #666;'>Powered by Ollama Vision Models</p>", unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("🎮 Controls")
        
        selected_model = st.selectbox(
            "🤖 Select Vision Model",
            get_available_models(),
            index=0,
        )
        
        format_type = st.selectbox(
            "📄 Output Format",
            ["markdown", "text", "json", "structured", "key_value"],
            help="Choose how you want the extracted text to be formatted"
        )

        max_workers = st.slider(
            "🔄 Parallel Processing",
            min_value=1,
            max_value=8,
            value=2,
            help="Number of images to process in parallel (for batch processing)"
        )

        enable_preprocessing = st.checkbox(
            "🔍 Enable Preprocessing",
            value=True,
            help="Apply image enhancement and preprocessing"
        )
        
        st.markdown("---")
        
        # Model info box
        if selected_model == "llava:7b":
            st.info("LLaVA 7B: Efficient vision-language model optimized for real-time processing")
        else:
            st.info("Llama 3.2 Vision: Advanced model with high accuracy for complex text extraction")

    # Initialize OCR Processor
    processor = OCRProcessor(model_name=selected_model, max_workers=max_workers)

    # Main content area with tabs
    tab1, tab2 = st.tabs(["📸 Image Processing", "ℹ️ About"])
    
    with tab1:
        # File upload area with multiple file support
        uploaded_files = st.file_uploader(
            "Drop your images here",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'pdf'],
            accept_multiple_files=True,
            help="Supported formats: PNG, JPG, JPEG, TIFF, BMP, PDF"
        )

        if uploaded_files:
            # Create a temporary directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                image_paths = []
                
                # Save uploaded files and collect paths
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    image_paths.append(temp_path)

                # Display images in a gallery
                st.subheader(f"📸 Input Images ({len(uploaded_files)} files)")
                cols = st.columns(min(len(uploaded_files), 4))
                for idx, uploaded_file in enumerate(uploaded_files):
                    with cols[idx % 4]:
                        image = Image.open(uploaded_file)
                        st.image(image, use_container_width=True, caption=uploaded_file.name)

                # Process button
                if st.button("🚀 Process Images"):
                    with st.spinner("Processing images..."):
                        if len(image_paths) == 1:
                            # Single image processing
                            result = process_single_image(
                                processor, 
                                image_paths[0], 
                                format_type,
                                enable_preprocessing
                            )
                            st.subheader("📝 Extracted Text")
                            st.markdown(result)
                            
                            # Download button for single result
                            st.download_button(
                                "📥 Download Result",
                                result,
                                file_name=f"ocr_result.{format_type}",
                                mime="text/plain"
                            )
                        else:
                            # Batch processing
                            results = process_batch_images(
                                processor,
                                image_paths,
                                format_type,
                                enable_preprocessing
                            )
                            
                            # Display statistics
                            st.subheader("📊 Processing Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Images", results['statistics']['total'])
                            with col2:
                                st.metric("Successful", results['statistics']['successful'])
                            with col3:
                                st.metric("Failed", results['statistics']['failed'])

                            # Display results
                            st.subheader("📝 Extracted Text")
                            for file_path, text in results['results'].items():
                                with st.expander(f"Result: {os.path.basename(file_path)}"):
                                    st.markdown(text)

                            # Display errors if any
                            if results['errors']:
                                st.error("⚠️ Some files had errors:")
                                for file_path, error in results['errors'].items():
                                    st.warning(f"{os.path.basename(file_path)}: {error}")

                            # Download all results as JSON
                            if st.button("📥 Download All Results"):
                                json_results = json.dumps(results, indent=2)
                                st.download_button(
                                    "📥 Download Results JSON",
                                    json_results,
                                    file_name="ocr_results.json",
                                    mime="application/json"
                                )

    with tab2:
        st.header("About Vision OCR Lab")
        st.markdown("""
        This application uses state-of-the-art vision language models through Ollama to extract text from images.
        
        ### Features:
        - 🖼️ Support for multiple image formats
        - 📦 Batch processing capability
        - 🔄 Parallel processing
        - 🔍 Image preprocessing and enhancement
        - 📊 Multiple output formats
        - 📥 Easy result download
        
        ### Models:
        - **LLaVA 7B**: Efficient vision-language model for real-time processing
        - **Llama 3.2 Vision**: Advanced model with high accuracy for complex documents
        """)

if __name__ == "__main__":
    main()