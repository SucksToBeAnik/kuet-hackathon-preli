# Mofa's Kitchen Buddy

documentation: http://localhost:8000/docs

### How to run the project

1. Install Ollama locally

2. Pull llama3.2 model

```bash
ollama pull llama3.2:1b
```

3. Pull nomic-embed-text model

```bash
ollama pull nomic-embed-text
```

4. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

5. Install dependencies

```bash
pip install -r requirements.txt
```

5. Run the project

```bash
fastapi dev main.py
```

### Current Routes

1. /ingredients - Get all ingredients
2. /ingredients - Create an ingredient
3. /files - Upload .text file containing recipes, currently not storing the file in database
4. /query - Chat with the llm based on the recipes in text file and your available ingredients

> Note: A recipe.txt file is provided in the root directory.
