import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Chain:
    def __init__(self):
        # Initialize the LLM with the Groq API key and chosen model
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        # Define the prompt to extract job postings from the scraped data
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills`, and `description`.
            Only return valid JSON.

            ### VALID JSON (NO PREAMBLE):
            """
        )
        # Combine the prompt with the language model
        chain_extract = prompt_extract | self.llm
        # Invoke the chain to generate a response
        res = chain_extract.invoke(input={"page_data": cleaned_text})

        try:
            # Parse the output into JSON format
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        
        # Return the result as a list (if it's a single JSON object, wrap it in a list)
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        # Define the prompt to write a cover letter based on job description and portfolio links
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Monish, a data scientist with expertise in analyzing data, developing machine learning models, and creating data-driven solutions to optimize business processes. Over your career, you have worked on various projects helping organizations harness the power of data to drive efficiency, scalability, and decision-making. 

            Your job is to write a cover letter to apply for the job mentioned above, describing your capability in fulfilling their needs based on your skills and experience. 

            Also, feel free to mention any relevant links or projects in your portfolio that align with the job description.
             
            Remember, you are Monish, a data scientist.  
            Mind the spacing of the cover letter 
            Do not provide a preamble.

            ### COVER LETTER (NO PREAMBLE):
            """
        )
        # Combine the prompt with the language model
        chain_email = prompt_email | self.llm
        # Invoke the chain to generate the email content
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        
        # Return the generated cover letter
        return res.content

if __name__ == "__main__":
    # For debugging purposes, print the Groq API key (ensure this is properly loaded)
    print(os.getenv("GROQ_API_KEY"))
