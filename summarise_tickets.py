import os
import PyPDF2
import openai
from openai import OpenAI
from typing import List
from fpdf import FPDF

def extract_text_from_pdfs(folder_path: str) -> List[str]:
    documents = []
    
    for filename in os.listdir(folder_path):
        # Ensure the file is a PDF file (you can add more file format checks if needed)
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as file:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                # Extract text from each page using list comprehension
                pages_text = [page.extract_text() for page in pdf_reader.pages]
                # Join the text from all pages into a single string
                pdf_text = "\n".join(pages_text)
                # Append the extracted text to the list
                documents.append(pdf_text)
    
    return documents

client = OpenAI(
    api_key="",
)

def summarize(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""You are an assistant tasked with summarizing forum discussions between customers \
            and support teams about technical issues. Each discussion includes multiple exchanges and may contain irrelevant \
            information. Your summary should focus on three main parts:  \
            1. **Issue Description:** Summarize the main problem or issue reported by the customer. \
            2. **Explanation of the Issue:** Summarize the support team's explanation of the issue, including any technical details provided. \
            3. **Resolution:** Summarize the solution or resolution provided by the support team, including any steps the customer needs to take. \
            Here is the discussion: \
            ---{text} ---
            Please provide a clear and concise summary with the following structure: \
            **Issue Description:**\
            [summary of the issue]\
            **Explanation of the Issue:**\
            [summary of the support team's explanation]\
            **Resolution:**\
            [summary of the resolution or solution]\
            Be sure to focus only on the relevant parts and ignore any unrelated information.\
            You can add these informations at the end: \
            'SAP version: [SAP version]', 'Attachments: [Attachments]', 'Defect number: [Defect number]'
            Reporter version: [ Reporter version]', SAP version: [ SAP version]"""}
        ],
    )
    return response.choices[0].message.content.strip()

def save_summary_as_pdf(summary: str, output_folder: str, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in summary.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(os.path.join(output_folder, filename))



def main():
    folder_path = "/Users/harouneaaffoute/Documents/OpenAI/tickets_for_AI_uncov" 
    output_folder = "/Users/harouneaaffoute/Documents/OpenAI/summarized_tickets"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    documents = extract_text_from_pdfs(folder_path)

    for idx, document in enumerate(documents):
        summary = summarize(document)
        print(f"Summary for Document {idx+1}:\n{summary}\n")
        save_summary_as_pdf(summary, output_folder, f"summary_{idx+1}.pdf")

if __name__ == "__main__":
    main()
