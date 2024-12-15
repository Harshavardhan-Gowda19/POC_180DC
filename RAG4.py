import os
import pyarrow as pa
import lancedb
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from lance.vector import PgVector
from lance.models import Agent
from lance.knowledge import PDFUrlKnowledgeBase
from lance.search import SearchType
from lance.llms import OpenAIChat
from dotenv import load_dotenv
import requests

# Initialize LanceDB
db_path = "lance_db"
db = lancedb.connect(db_path)

# Define the schema for the LanceDB table
schema = pa.schema([
    pa.field("id", pa.int32()),
    pa.field("filename", pa.string()),
    pa.field("content", pa.string()),
    pa.field("embedding", pa.list_(pa.float32())),
])

# Function to preprocess HTML files
def read_html(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        return soup.get_text()

# Function to preprocess PDF files
def read_pdf(file_path):
    try:
        pdf_reader = PdfReader(file_path)
        text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

# Initialize the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Folder path containing the files
folder_path = "Interview questions"

if not os.path.exists(folder_path):
    print(f"Folder does not exist: {folder_path}")
else:
    print(f"Folder exists: {folder_path}")

# Generator to create batches of data
def make_batches():
    file_counter = 0
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            print(f"Processing file: {file_name}")  # Debugging

            if file_name.endswith(".pdf") or file_name.endswith(".html"):
                try:
                    # Extract content based on file type
                    if file_name.endswith(".pdf"):
                        content = read_pdf(file_path)
                    elif file_name.endswith(".html"):
                        content = read_html(file_path)

                    if content.strip():  # Skip empty content
                        # Generate embedding
                        embedding = model.encode(content).tolist()

                        # Yield a record batch
                        yield pa.RecordBatch.from_arrays(
                            [
                                pa.array([file_counter]),
                                pa.array([file_name]),
                                pa.array([content]),
                                pa.array([embedding], type=pa.list_(pa.float32())),
                            ],
                            schema=schema,
                        )
                        print(f"Yielded batch for file: {file_name}")
                        file_counter += 1
                    else:
                        print(f"Skipping empty file: {file_name}")
                except Exception as e:
                    print(f"Failed to process {file_name}: {e}")

# Create the table and insert data in batches
table_name = "documents"
try:
    db.create_table(table_name, make_batches(), schema=schema)
    print("Table created and data inserted successfully!")
except Exception as e:
    print(f"Error creating table or inserting data: {e}")

class MainAgent:
    def __init__(self, applicant_id):
        load_dotenv()
        self.MY_BEARER_KEY = os.getenv('MY_BEARER_KEY')
        self.EXA_API_KEY = os.getenv('EXA_API_KEY')
        self.url = 'http://localhost:1337/api/queries'
        self.headers = {
            'Authorization': f'Bearer {self.MY_BEARER_KEY}'
        }
        
        self.params = {
            'filters[applicant_detail][id]': applicant_id
        }
        
        self.agent = Agent(
            model=Ollama(id="llama3.2"),
            tools=[ExaTools(api_key=self.EXA_API_KEY, num_results=10)],
            show_tool_calls=True,
            markdown=True
        )
        
        self.question_agent = Agent(
            model=Ollama(id="llama3.2"),
            markdown=True
        )
        
        self.queries_list = []
        self.responses = []

    def fetch_queries(self):
        """Fetch queries from the Strapi API and store them for processing."""
        max_retries = 5

        for attempt in range(max_retries):
            try:
                response = requests.get(self.url, params=self.params, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and isinstance(data['data'], list):
                        self.queries_list = [query['Queries'] for query in data['data'] if 'Queries' in query]
                        if not self.queries_list:
                            print("No valid queries found in the response.")
                        else:
                            print("Queries retrieved successfully.")
                    else:
                        print("Unexpected response format:", data)
                    return
                
                else:
                    print(f"Failed to retrieve queries. Status code: {response.status_code}")
                    print(response.json())
            
            except Exception as e:
                print(f"An error occurred during fetching queries: {e}")
            
            print(f"Retrying... ({attempt + 1}/{max_retries})")
        
        print("Failed to fetch queries after multiple attempts.") 

    def Agentic_Rag(self):
        """Retrieve responses using the Agentic RAG Handler."""
        knowledge_base = db.open_table(table_name)
        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            knowledge=knowledge_base,
            search_knowledge=True,
            markdown=True,
            show_tool_calls=True,
        )

        responses = []
        for query in self.queries_list:
            try:
                print(f"Processing query: {query}")
                run = agent.run(
                    f"Retrieve detailed information about: '{query}'. Use examples from the knowledge base for clarity."
                )
                if run.content:
                    responses.append(run.content)
                else:
                    print(f"No content retrieved for query: {query}")
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
        return responses

    def Web_Agent(self):
        """Retrieve comprehensive information for each query."""
        responses = []
        if not self.queries_list:
            print("No queries available to process.")
            return []

        for query in self.queries_list:
            try:
                print(f"Processing query: {query}")
                run: RunResponse = self.agent.run(
                    f"Provide a comprehensive overview of the following query: '{query}'. "
                    "Include detailed examples, key concepts, and relevant insights. "
                    "Ensure the content is rich and extensive; do not limit the information provided."
                )
                response = run.content

                if response:
                    responses.append(response)
                    print(f"Query processed successfully: {query}")
                else:
                    print(f"No content returned for query: '{query}'")

            except Exception as e:
                print(f"Error processing query '{query}': {e}")

        if not responses:
            print("No valid responses received.")
        else:
            print("Responses from agent:", responses)

        self.responses = responses
        return responses
    

    def combine_responses(self, rag_responses, qa_responses):
        """Combine RAG and Web Query responses using GPT model."""
        combined_prompt = (
            "The following are two sets of responses for the same queries. "
            "First set of responses comes from a knowledge base (RAG), and the second from web processing. "
            "Combine these responses into a single comprehensive answer for each query. Ensure clarity, cohesion, and avoid redundancy.\n\n"
        )

        for idx, (rag_response, qa_response) in enumerate(zip(rag_responses, qa_responses)):
            combined_prompt += f"Query {idx + 1}:\nRAG Response: {rag_response}\nWeb Response: {qa_response}\n\n"

        gpt_agent = Agent(model=OpenAIChat(id="gpt-4o"))
        try:
            combined_result = gpt_agent.run(combined_prompt)
            return combined_result.content if combined_result.content else "Failed to generate combined responses."
        except Exception as e:
            print(f"Error combining responses: {e}")
            return ""

    def generate_questions_from_combined(self, combined_response):
        """Generate insightful questions from the combined response."""
        if not combined_response:
            print("No combined response available to generate questions.")
            return []

        question_prompt = (
            f"Based on the following rich combined content: {combined_response}, "
            f"generate at least 20 insightful questions that promote deeper exploration of the subjects. "
            "Do not include answers; only provide the questions."
        )

        gpt_agent = Agent(model=Ollama(id="llama3.2"))
        try:
            run: RunResponse = gpt_agent.run(question_prompt)
            if run.content:
                print("Generated Questions:", run.content)
                return run.content
            else:
                print("No questions were generated.")
                return []
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []

def main():
    applicant_id = input("Enter applicant_id: ")
    
    # Fetch Queries
    main_agent = MainAgent(applicant_id)
    main_agent.fetch_queries()
    if not main_agent.queries_list:
        print("No queries found to process.")
        return

    # Agentic RAG for responses
    rag_responses = main_agent.Agentic_Rag()

    if rag_responses:
        print("RAG Handler Responses:")
        for response in rag_responses:
            print(response)
    else:
        print("No responses retrieved from the RAG handler.")

    # Web Agent for additional responses
    qa_responses = main_agent.Web_Agent()

    if qa_responses:
        print("WebAgent Responses:")
        for response in qa_responses:
            print(response)
    else:
        print("No responses retrieved from the WebAgent.")

    # Combine responses
    if rag_responses and qa_responses:
        combined_response = main_agent.combine_responses(rag_responses, qa_responses)
        print("\nCombined Responses:")
        print(combined_response)

        # Generate questions from combined responses
        questions = main_agent.generate_questions_from_combined(combined_response)
        print("\nGenerated Questions:")
        print(questions)
    else:
        print("Could not combine responses as one or both sets are missing.")

if __name__ == "__main__":
    main()
