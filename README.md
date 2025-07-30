# AI Chat Assistant

A sophisticated chatbot powered by Google Gemini and Perplexity AI with PDF document processing capabilities.

## Features

- 🤖 **Dual AI Response**: Get perspectives from both Gemini 2.5 Pro and Perplexity Sonar
- 📄 **PDF Processing**: Upload and analyze PDF documents with intelligent text extraction
- 🔍 **Semantic Search**: Advanced FAISS-based similarity search for document context
- 💬 **Chat History**: Persistent conversation history with file attachments
- 🎨 **Modern UI**: Clean, responsive Streamlit interface with custom styling

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Perplexity AI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-chat-assistant.git
cd ai-chat-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
   - Create a `.streamlit/secrets.toml` file
   - Add your API keys:
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
PERPLEXITY_API_KEY = "your_perplexity_api_key_here"
GOOGLE_API_KEY = "your_google_api_key_here"
```

## Usage

Run the application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## API Keys Setup

### Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `secrets.toml` file

### Perplexity AI API Key
1. Visit [Perplexity AI Settings](https://www.perplexity.ai/settings/api)
2. Generate an API key
3. Add it to your `secrets.toml` file

## File Structure

```
ai-chat-assistant/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore           # Git ignore rules
└── .streamlit/
    └── secrets.toml     # API keys (not tracked in git)
```

## Dependencies

- `streamlit` - Web app framework
- `PyPDF2` - PDF text extraction
- `numpy` - Numerical operations
- `faiss-cpu` - Similarity search
- `langchain-google-genai` - Google AI embeddings
- `google-generativeai` - Gemini AI API
- `requests` - HTTP requests

## Security

- Never commit your `secrets.toml` file to version control
- Keep your API keys private and secure
- The `.gitignore` file excludes sensitive files from git tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
