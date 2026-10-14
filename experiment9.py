
!pip -q install langchain-google-genai

import os
import json
from google.colab import userdata
from langchain_google_genai import ChatGoogleGenerativeAI

# Set API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0
)

# JSON prompt
prompt = """
Generate a JSON array containing exactly 3 books.
Each object must have these keys: title, author, year.
The year must be an integer.
Output only valid JSON, without Markdown or extra text.
"""

# Generate and validate JSON, retrying if necessary
for attempt in range(2):
    response = llm.invoke(prompt).text.strip()

    try:
        books = json.loads(response)

        assert isinstance(books, list) and len(books) == 3
        assert all(
            isinstance(book, dict)
            and set(book.keys()) == {"title", "author", "year"}
            and isinstance(book["title"], str)
            and isinstance(book["author"], str)
            and isinstance(book["year"], int)
            and not isinstance(book["year"], bool)
            for book in books
        )

        print("Valid JSON:")
        print(json.dumps(books, indent=2))
        break

    except (json.JSONDecodeError, AssertionError) as e:
        print("Validation failed:", e)
        prompt = """
        Output ONLY a valid JSON array of exactly 3 books.
        Each object must contain title (string), author (string),
        and year (integer). No Markdown or extra text.
        """
else:
    print("Failed to generate valid JSON after retry.")
