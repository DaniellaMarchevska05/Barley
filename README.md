# Barley - Your Cocktail Assistant 🍹

Welcome to **Barley**, a Python-based chat application that integrates with a large language model (LLM) to create a **Retrieval-Augmented Generation (RAG)** system. Barley is your virtual bartender, designed to provide cocktail recommendations, recipes, and mixology tips in a fun and engaging way. The application uses a vector database (FAISS) to store and retrieve cocktail data, and it leverages OpenAI's GPT-4 model for natural language understanding and generation.

---

## Features ✨

- **Cocktail Recommendations**: Get personalized cocktail suggestions based on your preferences.
- **Recipe Retrieval**: Access detailed recipes for a wide variety of cocktails.
- **User Preferences**: Barley remembers your favorite ingredients and cocktails for future recommendations.
- **Engaging Chat Interface**: A friendly and interactive chat interface with a playful tone, emojis, and animations.
- **Retrieval-Augmented Generation (RAG)**: Combines the power of LLMs with a vector database for accurate and context-aware responses.

---

## How It Works 🛠️

1. **Data Ingestion**: The application uses a dataset of cocktails from Kaggle, which is processed and stored in a vector database (FAISS) using OpenAI embeddings.
2. **User Interaction**: When you chat with Barley, your message is processed by the LLM to detect preferences and generate a response.
3. **Retrieval-Augmented Generation**: The system retrieves relevant cocktail information from the vector database and combines it with the LLM's generative capabilities to provide accurate and engaging responses.
4. **User Preferences**: Barley detects and stores your preferences (e.g., favorite ingredients or cocktails) to personalize future recommendations.

---

## Results and Thought Process 🧠

### Results
- **Accurate Recommendations**: The RAG system ensures that Barley provides accurate and relevant cocktail recommendations by combining the strengths of retrieval-based and generative models.
- **Personalization**: By detecting and storing user preferences, Barley can offer tailored suggestions, enhancing the user experience.
- **Engaging Experience**: The playful tone, emojis, and animations make interacting with Barley fun and immersive.

### Thought Process
- **Why RAG?**: Traditional LLMs can sometimes generate inaccurate or irrelevant responses. By integrating a retrieval system, Barley ensures that responses are grounded in accurate data from the cocktail dataset.
- **User Preferences**: Detecting and storing user preferences allows Barley to provide personalized recommendations, making the experience more engaging and useful.
- **UI/UX Design**: The chat interface is designed to be visually appealing and interactive, with animations and a modern design to enhance user engagement.

---

## How to Run the Project Locally 🚀

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (set as an environment variable or in a `.env` file)
- Kaggle API key (for downloading the dataset)

### Steps


1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/barley-cocktail-assistant.git
   cd barley-cocktail-assistant
1. **Create a .env file in the root directory and add your OpenAI API key:**
   ```bash
   OPENAI_API_KEY=your_openai_api_key
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
1. **Run the Application**
   Start the FastAPI server
   ```bash
   python main.py
  In a new terminal, start a local HTTP server to serve the frontend:
   ```bash
   python -m http.server 8080
