import streamlit as st
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import asyncio
from dotenv import load_dotenv
import PyPDF2  # For reading PDF files
import docx  # For reading DOCX files
import io  # To handle file-like objects for PyPDF2 and docx
import nest_asyncio  # Import nest_asyncio

# Apply nest_asyncio to allow nested event loops in Streamlit's environment
nest_asyncio.apply()

# --- Configuration and Setup ---

# Load environment variables for local development (optional, Streamlit Cloud uses st.secrets)
load_dotenv()

# Get Gemini API key
# Prioritize Streamlit secrets for deployment, fall back to environment variable for local
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

if not GEMINI_API_KEY:
    st.error("Gemini API Key not found. Please set it in `.streamlit/secrets.toml` or as an environment variable.")
    st.stop()  # Stop the app if API key is missing

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)


# Use st.cache_resource to load the NLP model only once
@st.cache_resource
def load_nlp_model():
    """Loads the Sentence Transformer model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading SentenceTransformer model: {e}")
        return None


nlp_model = load_nlp_model()

if nlp_model is None:
    st.stop()  # Stop the app if NLP model failed to load


# --- Helper Functions for File Reading ---

def read_pdf_text(file):
    """Reads text from a PDF file."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() or ""  # Use .extract_text() and handle None
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None
    return text


def read_docx_text(file):
    """Reads text from a DOCX file."""
    text = ""
    try:
        document = docx.Document(file)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return None
    return text


# --- Helper Function for Gemini API Calls ---
async def get_gemini_response(prompt, retries=3, delay=1.0):
    """
    Calls the Gemini API with exponential backoff, ensuring it runs asynchronously.
    """
    for i in range(retries):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            # Use asyncio.to_thread to run the potentially blocking/synchronous call
            # in a separate thread, making it awaitable for asyncio.gather
            response = await asyncio.to_thread(model.generate_content, prompt)
            return response.text
        except Exception as e:
            st.warning(f"Gemini API call failed (attempt {i + 1}/{retries}): {e}")
            if i < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
            else:
                st.error(f"Failed to get response from Gemini API after {retries} retries.")
                raise  # Re-raise the exception if all retries fail
    return "Summary generation failed."  # Should ideally not be reached


# --- Streamlit UI ---

st.set_page_config(page_title="Candidate Recommendation Engine", layout="centered")

# Custom CSS for a beautiful and colorful look with dark text
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #263238; /* Dark blue for general text */
    }
    .stApp {
        background: linear-gradient(135deg, #e0f2f7 0%, #c8e6c9 100%); /* Light, refreshing gradient background */
    }
    .stButton>button {
        background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%); /* Deep purple to bright blue gradient */
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2); /* Stronger shadow */
        transition: all 0.3s ease-in-out;
        width: 100%; /* Make button full width */
        margin-top: 1.5rem;
        border: none; /* Remove default border */
    }
    .stButton>button:hover {
        transform: translateY(-3px); /* Lift effect */
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.3); /* Even stronger shadow on hover */
    }
    /* Styling for the Job Description text area input (first one) */
    .stTextArea:nth-of-type(1) textarea {
        border-radius: 0.75rem;
        border: 2px solid #a7d9f7;
        padding: 1rem;
        background-color: #ffffff; /* White background */
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        transition: border-color 0.3s ease-in-out;
        color: #000000; /* Black text for job description */
    }
    /* Styling for the Resume text area input (second one) */
    .stTextArea:nth-of-type(2) textarea {
        border-radius: 0.75rem;
        border: 2px solid #a7d9f7;
        padding: 1rem;
        background-color: #ffffff; /* White background */
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        transition: border-color 0.3s ease-in-out;
        color: #ffffff; /* White text for resume input */
    }
    .stTextArea textarea:focus {
        border-color: #4a90e2; /* Highlight on focus */
        outline: none;
    }
    /* Specific styling for the file uploader text */
    .stFileUploader .st-emotion-cache-1c7y2ql p { /* Targets the text inside the file uploader */
        color: white; /* Make file uploader text white */
    }
    .stFileUploader label {
        font-weight: 700; /* Bolder label */
        color: #334155; /* Darker text */
        margin-bottom: 0.5rem;
        display: block;
    }
    .stFileUploader > div > button { /* Style for the file uploader button */
        background-color: #81c784; /* Greenish tone */
        color: #334155; /* Darker text */
        border: 1px solid #66bb6a;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s;
        width: auto; /* Override full width from general button style */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stFileUploader > div > button:hover {
        background-color: #66bb6a;
        transform: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .st-emotion-cache-1r4qj8v { /* Target the main content block for padding */
        padding-top: 2.5rem; /* More top padding */
        padding-bottom: 2.5rem; /* More bottom padding */
    }
    .st-emotion-cache-z5fcl4 { /* Target the main block container */
        background-color: #ffffff;
        border-radius: 1.5rem; /* More rounded */
        box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.2), 0 8px 15px -5px rgba(0, 0, 0, 0.1); /* Deeper shadow */
        padding: 3rem; /* More internal padding */
        margin-top: 2.5rem;
    }
    h1 {
        color: #1A237E; /* Dark blue for main title */
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2.8rem; /* Larger title */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    h2 { /* Streamlit's default h2 */
        color: #1A237E; /* Dark blue for H2 headings */
    }
    h3 {
        color: #1A237E; /* Dark blue for H3 headings */
        margin-top: 2.5rem;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid #e0e0e0; /* Thicker separator */
        padding-bottom: 0.75rem;
    }
    .st-emotion-cache-10o4u0s { /* Target st.info/st.warning boxes */
        background-color: #e3f2fd; /* Light blue info */
        border-left: 5px solid #2196f3; /* Blue left border */
        color: #1A237E; /* Dark blue for info text */
        border-radius: 0.5rem;
    }
    .st-emotion-cache-10o4u0s.st-emotion-cache-10o4u0s { /* Specific for warning */
        background-color: #fff3e0; /* Light orange warning */
        border-left: 5px solid #ff9800; /* Orange left border */
        color: #1A237E; /* Dark blue for warning text */
    }
    .candidate-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    .summary-box {
        background: linear-gradient(45deg, #e8f5e9, #e0f2f7); /* Subtle green-blue gradient */
        padding: 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #b3e5fc; /* Muted blue border */
        margin-top: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .summary-text {
        color: #263238; /* Darker text for readability */
        line-height: 1.7;
    }
    .st-emotion-cache-1wb00f0 { /* Target expander header */
        background-color: #f0f4c3; /* Light yellow for expander header */
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #cddc39;
        color: #1A237E; /* Dark blue for expander header text */
        /* Added for text wrapping in expander header */
        white-space: normal;
        overflow-wrap: break-word;
        margin-bottom: 0.5rem; /* Add some space below the header */
    }
    .st-emotion-cache-1wb00f0:hover {
        background-color: #e6ee9c; /* Slightly darker on hover */
    }
    /* Ensure all general text within the main content is dark */
    .st-emotion-cache-1r4qj8v p,
    .st-emotion-cache-1r4qj8v div,
    .st-emotion-cache-1r4qj8v span {
        color: #263238; /* Apply dark blue to general paragraphs, divs, and spans */
    }
    /* Override specific elements if needed, for example, the relevance score span */
    .st-emotion-cache-1r4qj8v strong {
        color: #263238; /* Ensure bold text is also dark */
    }
    .st-emotion-cache-1r4qj8v .st-emotion-cache-10o4u0s p { /* Info/warning box text */
        color: #1A237E; /* Dark blue for info/warning text */
    }
    </style>
""", unsafe_allow_html=True)

st.title("Candidate Recommendation Engine 🤖")
st.markdown("### Find the perfect match for your job with AI-powered insights. 🌟")

# --- Job Description Section ---
st.markdown("---")
st.header("1. Provide the Job Description 💼")
st.info("Paste the job description below. This will be used to understand the core requirements of the role.")
job_description = st.text_area(
    "Job Description Text:",
    height=200,
    placeholder="e.g., 'We are looking for a Senior Software Engineer with expertise in Python, Machine Learning, and cloud platforms like AWS. Strong communication skills and experience with agile methodologies are a plus.'"
)

# --- Candidate Resumes Section ---
st.markdown("---")
st.header("2. Upload Candidate Resumes 📄")
st.info(
    "You can paste multiple resumes (one per line) or upload `.txt`, `.pdf`, or `.docx` files. The app will extract text from all uploaded files.")

# Text area for pasted resumes - now with white text
candidate_resumes_text_area = st.text_area(
    "Paste candidate resumes here (one per line):",
    height=300,
    placeholder="e.g., 'Resume 1: John Doe, 5 years experience in Python, AI, ML, AWS. Excellent communicator.'\n'Resume 2: Jane Smith, 3 years experience in Java, web development, Azure. Team player.'"
)

# File uploader for multiple files
uploaded_files = st.file_uploader(
    "Or upload resume files (.txt, .pdf, .docx)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True
)

all_resumes_content = []

# Process uploaded files
if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        file_type = uploaded_file.type

        st.markdown(f"**Processing:** `{file_name}`")

        # Read file content based on type
        file_content = None
        if file_type == "text/plain":
            file_content = uploaded_file.getvalue().decode("utf-8")
        elif file_type == "application/pdf":
            file_content = read_pdf_text(io.BytesIO(uploaded_file.getvalue()))
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_content = read_docx_text(io.BytesIO(uploaded_file.getvalue()))
        else:
            st.warning(f"Unsupported file type for {file_name}: {file_type}. Skipping. ⚠️")
            continue

        if file_content:
            all_resumes_content.append({"text": file_content, "source_display": file_name})
            st.success(f"Successfully extracted text from `{file_name}`. ✅")
        else:
            st.error(f"Failed to extract text from `{file_name}`. Please check the file format. ❌")

# Combine pasted text and file content
if candidate_resumes_text_area:
    # Treat the entire pasted block as one "pasted resume" for simplicity in naming
    # If multiple distinct resumes are pasted, they should ideally be separated by the user
    all_resumes_content.append({"text": candidate_resumes_text_area, "source_display": "Pasted Text"})

final_resumes_list = all_resumes_content  # Now it's a list of dicts

# --- Recommendation Button ---
st.markdown("---")
if st.button("Find Best Candidates ✨", use_container_width=True):
    if not job_description:
        st.error("Please enter a job description to proceed. 🚨")
    elif not final_resumes_list:
        st.error("Please provide candidate resumes (paste or upload) to find recommendations. 📄")
    else:
        with st.spinner("Analyzing resumes and generating recommendations... This might take a moment! ⏳"):
            try:
                # Generate embedding for the job description
                job_embedding = nlp_model.encode([job_description])[0]

                candidates_raw_data = []
                # Prepare tasks for name extraction and embedding generation
                name_embedding_tasks = []
                for i, resume_item in enumerate(final_resumes_list):
                    resume_text = resume_item['text']
                    # Task for name extraction
                    name_prompt = f"Extract the full name of the person from the following resume text. Respond only with the name, or 'Unknown Candidate' if not found. Resume:\n\n{resume_text}"
                    name_embedding_tasks.append(get_gemini_response(name_prompt))

                    # Store data for later processing
                    candidates_raw_data.append({
                        "id": f"Candidate {i + 1}",  # Internal ID
                        "source_display": resume_item['source_display'],
                        "text": resume_text,
                        # Embedding will be generated after name extraction, or can be done here if no concurrency issue
                    })

                # Run all name extraction tasks concurrently
                name_results = asyncio.run(asyncio.gather(*name_embedding_tasks, return_exceptions=True))

                # Populate names and calculate similarities
                for i, candidate_data in enumerate(candidates_raw_data):
                    extracted_name = name_results[i]
                    if isinstance(extracted_name,
                                  Exception) or not extracted_name.strip() or extracted_name.lower() == 'unknown':
                        candidate_data["name"] = f"Candidate {candidate_data['id']}"  # Fallback
                    else:
                        candidate_data["name"] = extracted_name.strip()

                    # Generate embedding for the current resume
                    resume_embedding = nlp_model.encode([candidate_data["text"]])[0]

                    # Calculate similarity
                    similarity = cosine_similarity(job_embedding.reshape(1, -1), resume_embedding.reshape(1, -1))[0][0]
                    candidate_data["score"] = float(similarity)

                # Sort candidates by similarity score
                candidates_raw_data.sort(key=lambda x: x['score'], reverse=True)

                # Get top 5 candidates
                top_candidates = candidates_raw_data[:min(len(candidates_raw_data), 5)]  # Changed to 5

                # Prepare individual summary generation tasks for top 5
                individual_summary_tasks = []
                for candidate in top_candidates:
                    individual_summary_prompt = (
                        f"Given the job description:\n\"\"\"{job_description}\"\"\"\n\n"
                        f"And the candidate's resume:\n\"\"\"{candidate['text']}\"\"\"\n\n"
                        f"Generate a description (approximately 100 words) explaining why this person's resume fits this role. "
                        f"Highlight key skills, experiences, and qualifications from the resume that align with the job requirements. "
                        f"Start directly with the description, avoiding phrases like 'This resume shows...'"
                    )
                    individual_summary_tasks.append(get_gemini_response(individual_summary_prompt))

                # Run individual summary generation tasks concurrently
                individual_summaries_results = asyncio.run(
                    asyncio.gather(*individual_summary_tasks, return_exceptions=True))

                # --- Display Individual Candidate Recommendations ---
                st.markdown("---")
                st.header("Top Candidate Recommendations ✨")  # Removed "Individual"
                if top_candidates:
                    for i, candidate in enumerate(top_candidates):
                        display_description = individual_summaries_results[i]
                        if isinstance(display_description, Exception):
                            display_description = f"Error generating description: {display_description}"

                        # New output format: Name - Resume File - 100 word description
                        # Using a custom div to ensure text color and wrapping within the header
                        st.markdown(f"""
                            <div class="st-emotion-cache-1wb00f0">
                                <strong>Name:</strong> <span style="color: #1A237E;">{candidate['name']}</span> &nbsp;&nbsp;&nbsp; 
                                <strong>Source:</strong> <span style="color: #1A237E;">{candidate['source_display']}</span> &nbsp;&nbsp;&nbsp;
                                <strong>Relevance:</strong> <span style="color: #1A237E;">{candidate['score'] * 100:.2f}%</span>
                            </div>
                        """, unsafe_allow_html=True)
                        with st.expander(f"Click to view why {candidate['name']} fits this role",
                                         expanded=True):  # Expander text is now dynamic
                            st.markdown(f"""
                                <div class="summary-box">
                                    <p style="font-weight: 600; color: #1A237E; margin-bottom: 0.5rem;">Why This Resume Fits: 📝</p>
                                    <p class="summary-text">{display_description}</p>
                                </div>
                                <p style="margin-top: 1rem; font-weight: 600; color: #1A237E;">Original Resume Snippet:</p>
                                <div style="background-color: #f8fafc; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #e2e8f0; max-height: 200px; overflow-y: auto;">
                                    <pre style="white-space: pre-wrap; word-wrap: break-word; font-size: 0.85rem; color: #475569;">{candidate['text']}</pre>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info(
                        "No candidates found with relevant scores. Try adjusting your input or adding more resumes. 🔍")

            except Exception as e:
                st.error(f"An unexpected error occurred during processing: {e} 🚧")
                st.exception(e)  # Display full traceback for debugging