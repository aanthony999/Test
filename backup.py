import os
import logging
import threading
import time
import ssl
import smtplib
from email.message import EmailMessage
from datetime import datetime
import openpyxl
from openpyxl.styles import Font
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredFileLoader, Docx2txtLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import gradio as gr
from langchain.chains import create_extraction_chain
import dotenv
import tiktoken
from pymongo import MongoClient
import certifi
import gridfs
from bson.objectid import ObjectId
import pymongo
import json
import pandas as pd
import ast
import logging
import traceback
import re
import csv
from bson.objectid import ObjectId
from datetime import datetime



# Global variables and setup
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_URI += "&tlsAllowInvalidCertificates=true"
client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
client = MongoClient(MONGO_URI)

db = client['change_management_db']
content_collection = db['content_items']
fs = gridfs.GridFS(db)
logging.basicConfig(level=logging.DEBUG)

# Collections
plans_collection = db['communication_plans']
content_collection = db['content_items']

status_messages = []
scheduled_emails = []

llm = ChatOpenAI(temperature=0.7, model="gpt-4")

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Chroma setup
embeddings = OpenAIEmbeddings()
persist_directory = './chroma_db'
vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

# Pydantic model
class CommunicationPlan(BaseModel):
    full_plan: str = Field(description="A comprehensive communication plan including overview, timeline, audience-specific plans, and content plan")

# Utility functions
def add_documents_to_vectorstore(documents, collection_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    
    vectorstore.add_documents(documents=split_docs, collection_name=collection_name)
    print(f"Added {len(split_docs)} document chunks to the {collection_name} collection.")

def add_documents_to_comms(file_paths):
    documents = []
    for file_path in file_paths:
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            else:
                logging.warning(f"Unsupported file type: {file_path}")
                continue
            
            documents.extend(loader.load())
            logging.info(f"Successfully loaded file: {file_path}")
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {str(e)}")
    
    if documents:
        add_documents_to_vectorstore(documents, "comms_documents")
    else:
        logging.warning("No documents were successfully loaded.")

def get_retriever(collection_name):
    return vectorstore.as_retriever(
        search_kwargs={'k': 5},
        collection_name=collection_name
    )

def load_content_type_rules(file_path='content_type_rules.csv'):
    rules = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            content_type = row.pop('Content Type List')
            rules[content_type] = {k: v.lower() == 'yes' for k, v in row.items()}
    return rules

# Load the rules at the start of your script
content_type_rules = load_content_type_rules()

def save_content_to_database(content, project_name, item):
    try:
        # Ensure project_name is not empty
        if not project_name or project_name.strip() == "":
            logging.error(f"Attempted to save content with empty project name")
            return None

        content_data = {
            'project_name': project_name.strip(),  # Ensure it's stripped of whitespace
            'content_name': f"{item['Content Type']} - {item['Stakeholder Profile']} - {item['Campaign']}",
            'content': content,
            'stakeholder_profile': item['Stakeholder Profile'],
            'campaign': item['Campaign'],
            'key_messages': item['Key Messages'],
            'content_type': item['Content Type'],
            'channel': item['Channel'],
            'status': 'draft',
            'created_at': datetime.now()
        }
        result = content_collection.insert_one(content_data)
        logging.info(f"Saved content to database for project '{project_name}': {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logging.error(f"Error saving content to database: {str(e)}")
        return None

def process_uploaded_files(files):
    uploaded_content = []
    for file in files:
        try:
            content = file.read().decode('utf-8')
            uploaded_content.append(content)
        except Exception as e:
            logging.error(f"Error processing uploaded file {file.name}: {str(e)}")
    return "\n\n".join(uploaded_content)


#extract stakeholder profiles from uploaded file
def extract_stakeholder_profiles(files):
    extracted_profiles = []
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    for file in files:
        try:
            file_extension = os.path.splitext(file.name)[1].lower()
            
            if file_extension == '.txt':
                loader = TextLoader(file.name)
            elif file_extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file.name)
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file.name)
            else:
                print(f"Unsupported file type: {file.name}")
                continue
            
            docs = loader.load()
            content = ' '.join([doc.page_content for doc in docs])
            
            prompt = f"""
            Extract stakeholder profiles from the following document. For each stakeholder, provide the following information:
            1. Profile Name
            2. Areas of Interest / Values
            3. Areas of Concern
            4. Current State (if not explicitly stated, use 'Not specified')
            5. Future State (if not explicitly stated, use 'Not specified')
            6. Key Changes
            7. Negative Change Impact (Low, Medium, or High)
            8. Total Change Impact (Low, Medium, or High)
            9. Change Interest (Low, Medium, or High)
            10. Change Influence (Low, Medium, or High)
            11. Size (a number from 1-5, use 3 if not specified)
            12. Internal/External (Internal, External, or Both)

            Here's the document content:

            {content}

            Format the output as a JSON array of objects, where each object represents a stakeholder profile with the 12 elements listed above as key-value pairs.
            """

            response = llm.predict(prompt)
            
            try:
                # Try to parse as JSON first
                profiles = json.loads(response)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to evaluate as a Python literal
                try:
                    profiles = ast.literal_eval(response)
                except:
                    print(f"Error parsing LLM response for file {file.name}. Response: {response}")
                    continue

            if isinstance(profiles, list):
                for profile in profiles:
                    if isinstance(profile, dict) and len(profile) == 12:
                        profile_list = [
                            profile.get('Profile Name', ''),
                            profile.get('Areas of Interest / Values', ''),
                            profile.get('Areas of Concern', ''),
                            profile.get('Current State', 'Not specified'),
                            profile.get('Future State', 'Not specified'),
                            profile.get('Key Changes', ''),
                            profile.get('Negative Change Impact', 'Low'),
                            profile.get('Total Change Impact', 'Low'),
                            profile.get('Change Interest', 'Low'),
                            profile.get('Change Influence', 'Low'),
                            profile.get('Size', 3),
                            profile.get('Internal/External', 'Internal')
                        ]
                        extracted_profiles.append(profile_list)
            else:
                print(f"Unexpected response format for file {file.name}")

        except Exception as e:
            print(f"Error processing file {file.name}: {str(e)}")

    return extracted_profiles

def edit_content_with_ai(content, edit_request):
    chat = ChatOpenAI(temperature=0.7)
    messages = [
        SystemMessage(content="You are an AI assistant that helps edit and improve content. Your task is to modify the given content based on the user's request."),
        HumanMessage(content=f"Here's the original content:\n\n{content}\n\nEdit request: {edit_request}")
    ]
    response = chat(messages)
    return response.content

def send_email(sender, password, receiver, subject, body):
    try:
        em = EmailMessage()
        em['From'] = sender
        em['To'] = receiver
        em['Subject'] = subject
        em.set_content(body)

        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(sender, password)
            smtp.sendmail(sender, receiver, em.as_string())
        
        logging.info(f"Email sent successfully to {receiver}")
        return True
    except Exception as e:
        logging.error(f"Failed to send email: {str(e)}")
        return False

def email_sender_worker():
    global scheduled_emails
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        emails_to_send = [email for email in scheduled_emails if email[4] == current_time]
        
        for email in emails_to_send:
            try:
                content_item = content_collection.find_one({'_id': ObjectId(email[1])})
                if content_item:
                    content = content_item['content'].strip()
                    lines = content.split('\n')
                    subject = lines[0].strip()
                    body = '\n'.join(lines[1:]).strip()
                    
                    send_email(
                        sender='amalantony7799@gmail.com',
                        password='vktf ygjm xmja luon',
                        receiver=email[3],
                        subject=subject,
                        body=body
                    )
                    print(f"Email sent: {email}")
                    
                    # Update content status to 'delivered'
                    content_collection.update_one(
                        {'_id': ObjectId(email[1])},
                        {'$set': {'status': 'delivered'}}
                    )
                    
                    # Update status in scheduled_emails list
                    email[5] = 'delivered'
                    
                    scheduled_emails = [e for e in scheduled_emails if e != email]
                else:
                    print(f"Content not found for email: {email}")
            except Exception as e:
                print(f"Error sending email: {e}")
        
        time.sleep(60)  # Check every minute

# Core functions

def parse_plan_response(response_content):
    # Remove triple backticks and "python" or "json" if present
    response_content = re.sub(r'^```(?:python|json)\s*|\s*```$', '', response_content.strip())
    
    try:
        # Try to parse the entire response as a JSON array
        plan_data = json.loads(response_content)
        if isinstance(plan_data, list):
            return [sanitize_item(item) for item in plan_data]
    except json.JSONDecodeError:
        # If JSON parsing fails, fall back to line-by-line parsing
        plan_data = []
        current_item = {}
        for line in response_content.split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        plan_data.append(sanitize_item(item))
                        current_item = {}
                except json.JSONDecodeError:
                    pass
            elif ':' in line:
                key, value = map(str.strip, line.split(':', 1))
                if key in ['Stakeholder Profile', 'Campaign', 'Content ID', 'Key Messages', 'Content Type', 'Channel']:
                    if current_item:
                        plan_data.append(sanitize_item(current_item))
                        current_item = {}
                    current_item[key] = value
            elif line and current_item:
                last_key = list(current_item.keys())[-1]
                current_item[last_key] += ' ' + line

        if current_item:
            plan_data.append(sanitize_item(current_item))

    return plan_data

def sanitize_item(item):
    required_keys = ['Stakeholder Profile', 'Campaign', 'Content ID', 'Key Messages', 'Content Type', 'Channel']
    sanitized_item = {}
    for key in required_keys:
        value = item.get(key, 'Not specified')
        if key == 'Key Messages' and isinstance(value, str):
            # Split key messages by comma, removing any surrounding quotes
            value = [re.sub(r'^["\']\s*|\s*["\']$', '', msg.strip()) for msg in value.split(',')]
        sanitized_item[key] = value
    return sanitized_item

def list_to_dataframe(plan_data):
    if not plan_data:
        return pd.DataFrame(columns=['Stakeholder Profile', 'Campaign', 'Content ID', 'Key Messages', 'Content Type', 'Channel'])
    
    df = pd.DataFrame(plan_data)
    
    # Ensure all required columns are present
    required_columns = ['Stakeholder Profile', 'Campaign', 'Content ID', 'Key Messages', 'Content Type', 'Channel']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 'Not specified'
    
    return df[required_columns]

def generate_communications_plan(project_name, change_type, description, stakeholder_profiles, channels, market, industry, supporting_docs):
    try:
        status_messages = []
        final_plan = None
        content_plan = None

        status_messages.append("Processing uploaded documents...")
        if supporting_docs:
            file_paths = [doc.name for doc in supporting_docs]
            add_documents_to_comms(file_paths)
        else:
            status_messages.append("No supporting documents uploaded. Proceeding with basic information only.")

        status_messages.append("Retrieving relevant information from uploaded documents...")
        
        comms_retriever = get_retriever("comms_documents")
        
        retrieval_query = f"""
        Retrieve relevant information for a communication plan about:
        Project: {project_name}
        Change Type: {change_type}
        Description: {description}
        Market: {market}
        Industry: {industry}
        """
        
        relevant_docs = comms_retriever.get_relevant_documents(retrieval_query)
        relevant_info = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        if not relevant_info.strip():
            status_messages.append("No specific relevant information found in uploaded documents. Proceeding with general knowledge.")
            relevant_info = "No specific relevant information found. Using general knowledge for plan generation."

        status_messages.append("Generating communication plan...")
        
        # Convert stakeholder_profiles to the expected format
        formatted_profiles = []
        for profile in stakeholder_profiles:
            if isinstance(profile, list) and len(profile) >= 12:
                formatted_profiles.append({
                    "name": profile[0],
                    "areas_of_interest": profile[1],
                    "areas_of_concern": profile[2],
                    "current_state": profile[3],
                    "future_state": profile[4],
                    "key_changes": profile[5],
                    "negative_change_impact": profile[6],
                    "total_change_impact": profile[7],
                    "change_interest": profile[8],
                    "change_influence": profile[9],
                    "size": profile[10],
                    "internal_external": profile[11]
                })

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI communications planner specializing in change management. Create a detailed communications plan based on the following input and rules:

        Project Name: {project_name}
        Change Type: {change_type}
        Description: {description}
        Stakeholder Profiles: {stakeholder_profiles}
        Communication Channels: {channels}
        Market: {market}
        Industry: {industry}
        Relevant Context from uploaded documents: {relevant_info}
        content type rules: {content_type_rules}


        Output Rules:
        1. Generate a structured list of communication items for each stakeholder profile and each stakeholder profile can have multiple key messages and multiple campaigns.
        2. For each stakeholder and campaign combination, create multiple content items as needed, based on the stakeholder's characteristics and the campaign's requirements.
             a) For example, if the stakeholder is a high-impact executive with a high interest and influence in the change, you may need to create more content items for them.
             b) If the stakeholder is a low-impact executive with a low interest and influence in the change, you may need to create less content items for them.
             c) If the stakeholder is a high-impact frontline worker with a high interest and influence in the change, you may need to create more content items for them.
             d) If the stakeholder is a low-impact frontline worker with a low interest and influence in the change, you may need to create less content items for them.
             e) If the stakeholder is both high-impact and high-interest, you may need to create more content items for them.
             f) If the stakeholder is both low-impact and low-interest, you may need to create less content items for them.
        3. Each item should include: Stakeholder Profile, Campaign (Awareness, Desire, Knowledge, Ability, or Reinforcement), Content ID, Key Messages (multiple, comma-separated), Content Type, and Channel.
        4. Use the ADKAR model (Awareness, Desire, Knowledge, Ability, Reinforcement) for the campaigns.
        5. Ensure content is tailored to each stakeholder profile and relevant to the change project.
        6. Use a mix of general (G) and personalized (P) content across stakeholders and campaigns.
        7. Content ID should follow the format: <G_AW_1>, <G_AW_2>, etc. for general content, <P_AW_1>, <P_AW_2>, etc. for personalized content.
        8. Key Messages should be a list of brief statements, each max 20 words, separated by commas.
        9. Content Type should be appropriate for the campaign and stakeholder. You must adhere to the following rules when selecting content types for each campaign:
             a) Only suggest content types that are marked as 'Yes' for the specific campaign type (Awareness, Desire, Knowledge, Ability, or Reinforcement).
             b) Content types can be used in multiple campaigns if they are marked 'Yes' for those campaigns.
             c) Ensure that the content types suggested are appropriate for the campaign's goals and target audience.
        10. Channel should be selected from the provided communication channels.
        11. Consider the stakeholder's role, influence, and information needs when determining the number and types of communications for each campaign phase.
        12. For high-impact stakeholders (e.g., executives, key decision-makers), create more frequent and varied communications.
        13. Ensure a logical progression of communications within each campaign phase, building upon previous messages.

        Present the communication plan as a list of dictionaries, each representing a communication item.
        """)
        ])

        llm = ChatOpenAI(temperature=0.7, max_tokens=4000)
        chain = prompt | llm

        chain_input = {
            "project_name": project_name,
            "change_type": change_type,
            "description": description,
            "stakeholder_profiles": json.dumps(formatted_profiles),
            "channels": ', '.join(channels) if isinstance(channels, list) else channels,
            "market": market,
            "industry": industry,
            "relevant_info": relevant_info,
            "content_type_rules": json.dumps(content_type_rules)
        }

        response = chain.invoke(chain_input)
        if hasattr(response, 'content'):
            plan_content = response.content
        else:
            plan_content = str(response)
        
        logging.debug(f"Raw AI response: {plan_content}")

        # Parse the response into a list of dictionaries
        plan_data = parse_plan_response(plan_content)
        logging.debug(f"Parsed plan data: {plan_data}")

        if not plan_data:
            logging.error("Parsed plan data is empty")
            raise ValueError("Failed to parse the AI response into a valid plan.")

        plan_df = list_to_dataframe(plan_data)
        logging.debug(f"DataFrame: {plan_df}")

        if plan_df.empty:
            logging.error("Generated plan DataFrame is empty")
            raise ValueError("Generated plan DataFrame is empty.")

        return "Communication plan generated successfully.", plan_df
    except Exception as e:
        logging.error(f"Error in generate_plan_and_content: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return f"Error: {str(e)}", None

#generate content from plan function
def generate_content_for_plan(plan_df, current_project, project_name_input):
            effective_project_name = current_project if current_project else project_name_input.strip()
            if not effective_project_name:
                return "Error: No project name provided. Please enter a project name or load an existing plan."
            
            return generate_content_for_plan_impl(plan_df, effective_project_name)

def generate_content_for_plan_impl(plan_df, effective_project_name):
    if not isinstance(plan_df, pd.DataFrame) or plan_df.empty:
        return "Error: No valid communication plan found. Please generate or load a plan first."

    logging.info(f"Generating content for project: '{effective_project_name}'")
    logging.info(f"Number of items in plan: {len(plan_df)}")

    llm = ChatOpenAI(temperature=0.7)
    generated_content = []

    # Get the retriever for the uploaded documents
    comms_retriever = get_retriever("comms_documents")

    for _, row in plan_df.iterrows():
        
        # Construct a query to retrieve relevant context about the change
        retrieval_query = f"""
        Retrieve relevant information for:
        Project: {effective_project_name}
        Stakeholder: {row['Stakeholder Profile']}
        Campaign: {row['Campaign']}
        Content Type: {row['Content Type']}
        Key Messages: {', '.join(row['Key Messages'])}
        """

        # Get relevant documents
        relevant_docs = comms_retriever.get_relevant_documents(retrieval_query)
        relevant_context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in creating content for change management communications. 
            Create a single piece of content based on the following specifications:

            Project Name: {project_name}
            Stakeholder Profile: {stakeholder}
            Campaign: {campaign}
            Key Messages: {key_messages}
            Content Type: {content_type}
            Channel: {channel}
             
            relevant Context from uploaded documents:{relevant_context}
             
            Instructions:
            1. Understand that the Content Type represents the purpose of the communication within the ADKAR model (Awareness, Desire, Knowledge, Ability, Reinforcement).
            2. The Channel specifies the format or medium of delivery (e.g., email, blog post, Microsoft Sway presentation).
            3. Format the content appropriately for the specified Channel.
            4. Ensure the content fulfills the Content Type (purpose) and incorporates all key messages cohesively.
            5. Tailor the tone and language to the stakeholder profile and campaign stage.
            6. If the Channel is email, include a subject line as the first line and follow best practices for email writing.
            7. If the Channel is Microsoft Sway, include a title as the first line and follow best practices for Sway presentations.
            8. If the Channel is Blog, include a headline as the first line and follow best practices for blog writing.
            7. Create a unified piece of content that addresses all key messages effectively while achieving the Content Type's purpose.

            Generate the content now, starting with the title/subject (if applicable):"""),
            ("human", "Generate the content.")
        ])

        content = llm(prompt.format_messages(
            project_name=effective_project_name,
            stakeholder=row['Stakeholder Profile'],
            campaign=row['Campaign'],
            key_messages=", ".join(row['Key Messages']),
            content_type=row['Content Type'],
            channel=row['Channel'],
            relevant_context=relevant_context
        ))

        content_id = save_content_to_database(content.content, effective_project_name.strip(), row)
        if content_id:
            generated_content.append(content_id)
            logging.info(f"Generated and saved content for project '{effective_project_name}': {content_id}")
        else:
            logging.error(f"Failed to save content for project '{effective_project_name}'")

    logging.info(f"Total content items generated and saved for project '{effective_project_name}': {len(generated_content)}")
    return f"Content generated for project: {effective_project_name}"

def generate_content_for_row(row, project_name):
    llm = ChatOpenAI(temperature=0.7)
    comms_retriever = get_retriever("comms_documents")
    
    retrieval_query = f"""
    Retrieve relevant information for:
    Project: {project_name}
    Stakeholder: {row['Stakeholder Profile']}
    Campaign: {row['Campaign']}
    Content Type: {row['Content Type']}
    Key Messages: {', '.join(row['Key Messages'])}
    """
    
    relevant_docs = comms_retriever.get_relevant_documents(retrieval_query)
    relevant_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in creating content for change management communications. 
        Create a single piece of content based on the following specifications:

        Project Name: {project_name}
        Stakeholder Profile: {stakeholder}
        Campaign: {campaign}
        Key Messages: {key_messages}
        Content Type: {content_type}
        Channel: {channel}
         
        relevant Context from uploaded documents:{relevant_context}
         
        Instructions:
        1. Understand that the Content Type represents the purpose of the communication within the ADKAR model (Awareness, Desire, Knowledge, Ability, Reinforcement).
        2. The Channel specifies the format or medium of delivery (e.g., email, blog post, Microsoft Sway presentation).
        3. Format the content appropriately for the specified Channel.
        4. Ensure the content fulfills the Content Type (purpose) and incorporates all key messages cohesively.
        5. Tailor the tone and language to the stakeholder profile and campaign stage.
        6. If the Channel is email, include a subject line as the first line and follow best practices for email writing.
        7. If the Channel is Microsoft Sway, include a title as the first line and follow best practices for Sway presentations.
        8. If the Channel is Blog, include a headline as the first line and follow best practices for blog writing.
        9. Create a unified piece of content that addresses all key messages effectively while achieving the Content Type's purpose.

        Generate the content now, starting with the title/subject (if applicable):"""),
        ("human", "Generate the content.")
    ])
    
    content = llm(prompt.format_messages(
        project_name=project_name,
        stakeholder=row['Stakeholder Profile'],
        campaign=row['Campaign'],
        key_messages=", ".join(row['Key Messages']),
        content_type=row['Content Type'],
        channel=row['Channel'],
        relevant_context=relevant_context
    ))
    
    content_id = save_content_to_database(content.content, project_name.strip(), row)
    return content_id
def get_project_options():
    projects = content_collection.distinct('project_name')
    return sorted(projects)

def update_existing_content_status():
    try:
        result = content_collection.update_many(
            {"status": {"$exists": False}},
            {"$set": {"status": "draft"}}
        )
        print(f"Updated {result.modified_count} documents with default 'draft' status.")
    except Exception as e:
        print(f"Error updating existing content status: {str(e)}")

def save_plan_version(project_name, plan_df):
    """
    Save a new version of the plan to MongoDB.
    """
    plan_data = plan_df.to_dict('records')
    result = plans_collection.update_one(
        {'project_name': project_name},
        {
            '$push': {
                'versions': {
                    'plan': plan_data,
                    'timestamp': datetime.now()
                }
            },
            '$set': {'current_plan': plan_data}
        },
        upsert=True
    )
    logging.info(f"Saved new version of plan for project {project_name}")
    return result.modified_count > 0

def get_plan_changes(project_name):
    """
    Retrieve the changes between the last two versions of the plan.
    """
    project = plans_collection.find_one({'project_name': project_name})
    if not project or 'versions' not in project or len(project['versions']) < 2:
        logging.info(f"No previous version found for project {project_name}")
        return None, None

    current_version = project['versions'][-1]['plan']
    previous_version = project['versions'][-2]['plan']

    current_df = pd.DataFrame(current_version)
    previous_df = pd.DataFrame(previous_version)

    return previous_df, current_df

def identify_changed_rows(old_df, new_df):
    """
    Identify the rows that have changed between two versions of the plan.
    """
    changed_indices = []
    for idx, (old_row, new_row) in enumerate(zip(old_df.itertuples(index=False), new_df.itertuples(index=False))):
        if old_row != new_row:
            changed_indices.append(idx)
    
    # Check for added rows
    if len(new_df) > len(old_df):
        for idx in range(len(old_df), len(new_df)):
            changed_indices.append(idx)
    
    return changed_indices

def generate_content_for_changed_rows(project_name):
    """
    Generate content only for the rows that have changed in the plan.
    """
    old_df, new_df = get_plan_changes(project_name)
    if old_df is None or new_df is None:
        return "No changes detected or insufficient versions to compare."

    changed_indices = identify_changed_rows(old_df, new_df)
    if not changed_indices:
        return "No changes detected in the plan."

    generated_content = []
    for idx in changed_indices:
        row = new_df.iloc[idx]
        content_id = generate_content_for_row(row, project_name)
        if content_id:
            generated_content.append(content_id)

    return f"Content generated for {len(generated_content)} changed items in project: {project_name}"

# Gradio interface functions
def about_page():
    with gr.Blocks() as page:
        with gr.Column(elem_classes="about-background"):
            gr.Markdown("""        
            MetamorphAI is a tool designed to streamline and enhance change communication processes. It leverages generative AI to assist in planning, creating, and managing communications for organisational change initiatives.
            
            ## Features
            
            ### 1. Communications Strategy
            
            **What it does:** This feature generates a tailored communication plans, content plan, and the content in the background for your change initiatives.
            
            **How to use it:**
            - Enter project details, including name, change type, and description.
            - Build stakeholder profiles to personalise communications
            - Generate Communications Plan
            - you can upload any supporting documents related to your change initiative.
            - Click "Generate Communications Plan" to receive a comprehensive plan.
            - Click "Generate Content" to generate draft content specific to your plan.
            - You also have the ability to revisit other projects you are working on and make changes to them.
            
            ### 2. Content Manager
            
            **What it does:** Allows you to view, edit, and customise the generated content based on your communication plan.
            
            **How to use it:**
            - Select "Refresh Projects" to retrieve all the content generated
            - Select a project from the dropdown menu.
            - You can filter by campaign
            - Choose a specific content item to view.
            - Use the AI & text Editor to make changes or improvements to the content.
            - Save your edited content for future use.
            
            ### 3. Communications Delivery
            
            **What it does:** Enables you to schedule and manage the delivery of your change communication content.
            
            **How to use it:**
            - Select a project and choose the content you want to schedule.
            - Track your content delivery through the tags in the content title
            - Set the recipient email, date, and time for delivery.
            - Use the scheduling interface to plan and organize your communication timeline.

            
            This purpose of this POC is to demonstrate how we can enhance efficiency, consistency, and effectiveness in change communication processes, leveraging the power of Generative AI to support your organisational transformation initiatives.
            """)
    return page

def change_accelerator_page():
    with gr.Blocks() as page:
        gr.Markdown("Generate a tailored communications plan and content for your change initiative.")

        with gr.Tabs():
            with gr.TabItem("New Project"):
                with gr.Accordion("Enter Project Details", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            project_name = gr.Textbox(label="Project Name", placeholder="Enter project name")
                            change_type = gr.Dropdown(["Select change type", "Organizational", "Technological", "Process"], label="Change Type", value="Select change type")
                            market = gr.Textbox(label="Market", placeholder="Enter target market")
                            industry = gr.Textbox(label="Industry", placeholder="Enter industry")
                        
                        with gr.Column(scale=1):
                            description = gr.Textbox(label="Brief Description", lines=3, placeholder="Describe your change initiative")
                            channels = gr.CheckboxGroup(
                                ["Email", "Video", "In-person", "Intranet", "Microsoft Sway", "Blog"], 
                                label="Select Communication Channels",
                                info="Select all applicable channels"
                            )
                            supporting_docs = gr.File(label="Upload Supporting Documents", file_count="multiple")

                with gr.Accordion("Add Stakeholder Profiles", open=False):
                    stakeholder_profiles = gr.State([])
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            profile_name = gr.Textbox(label="Profile Name", placeholder="Enter stakeholder profile name")
                            areas_of_interest = gr.Textbox(label="Areas of Interest / Values", placeholder="Enter areas of interest")
                            areas_of_concern = gr.Textbox(label="Areas of Concern", placeholder="Enter areas of concern")
                            current_state = gr.Textbox(label="Current State", placeholder="Describe current state")
                            future_state = gr.Textbox(label="Future State", placeholder="Describe future state")
                            key_changes = gr.Textbox(label="Key Changes", placeholder="Enter key changes (Mandatory)")
                        
                        with gr.Column(scale=1):
                            negative_change_impact = gr.Dropdown(["Low", "Medium", "High"], label="Negative Change Impact", value="Low")
                            total_change_impact = gr.Dropdown(["Low", "Medium", "High"], label="Total Change Impact", value="Low")
                            change_interest = gr.Dropdown(["Low", "Medium", "High"], label="Change Interest", value="Low")
                            change_influence = gr.Dropdown(["Low", "Medium", "High"], label="Change Influence", value="Low")
                            size = gr.Number(label="Size (Enter group size)")
                            internal_external = gr.Dropdown(["Internal", "External", "Both"], label="Internal / External", value="Internal")

                    with gr.Row():
                        add_profile_btn = gr.Button("Add Profile", variant="secondary")
                        extract_profiles_btn = gr.Button("Extract Stakeholder Profiles from Uploaded Documents")

                    with gr.Accordion("View Stakeholder Profiles", open=True):
                        profile_list = gr.Dataframe(
                            headers=["Name", "Areas of Interest", "Areas of Concern", "Current State", "Future State", 
                                    "Key Changes", "Negative Impact", "Total Impact", "Interest", "Influence", "Size", "Internal/External"],
                            label="",
                            interactive=True
                        )
            
                        delete_profile_btn = gr.Button("Delete Profile (Most recent deleted first)", variant="secondary")

                with gr.Accordion("Generate & Edit Communications Plan", open=False):
                    generate_btn = gr.Button("Generate Communications Plan", variant="primary", size="lg")

                    new_project_plan_output = gr.Dataframe(
                        label="Communication Plan",
                        headers=["Stakeholder Profile", "Campaign", "Content ID", "Key Messages", "Content Type", "Channel"],
                        interactive=False,
                    )
                    
                    with gr.Group(visible=False) as new_project_edit_form:
                        gr.Markdown("### Edit Plan Item")
                        new_project_row_index = gr.Number(label="Row Index", precision=0)
                        new_project_stakeholder_profile = gr.Textbox(label="Stakeholder Profile")
                        new_project_campaign = gr.Textbox(label="Campaign")
                        new_project_content_id = gr.Textbox(label="Content ID")
                        new_project_key_messages = gr.Textbox(label="Key Messages")
                        new_project_content_type = gr.Textbox(label="Content Type")
                        new_project_channel = gr.Textbox(label="Channel")
                        new_project_update_row_btn = gr.Button("Update Row")
                    
                    new_project_save_changes_btn = gr.Button("Save All Changes", variant="primary", visible=False)
                    save_plan_btn = gr.Button("Save Plan", variant="primary", visible=False)

            with gr.TabItem("Plan Management & Content Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("Load & Manage Communications Plans", open=True):
                            refresh_projects_btn = gr.Button("Refresh Projects", variant="secondary")
                            existing_projects = gr.Dropdown(label="Select Existing Project", choices=[], interactive=True)
                            load_plan_btn = gr.Button("Load Plan", variant="secondary")
                            close_plan_btn = gr.Button("Close Loaded Plan", variant="secondary", visible=False)

                            gr.Markdown("""
                            Click refresh button first to see saved plans

                            Editing Instructions:
                            1. **Select a Row**: Click on any row in the table to edit its contents.
                            2. **Make Changes**: Update the information in the edit form that appears.
                            3. **Update Row**: Click 'Update Row' to apply changes to the selected row.
                            4. **Save All Changes**: After editing desired rows, click 'Save All Changes' to update the entire plan.
                            
                            Note: Changes are not permanent until you click 'Save All Changes'.
                            """)

                    with gr.Column(scale=2):
                        with gr.Accordion("Edit Plan & Generate Content", open=True):
                            existing_plan_output = gr.Dataframe(
                                label="Communication Plan",
                                headers=["Stakeholder Profile", "Campaign", "Content ID", "Key Messages", "Content Type", "Channel"],
                                interactive=False,
                            )
                            
                            with gr.Group(visible=False) as edit_form:
                                gr.Markdown("### Edit Plan Item")
                                row_index = gr.Number(label="Row Index", precision=0)
                                stakeholder_profile = gr.Textbox(label="Stakeholder Profile")
                                campaign = gr.Textbox(label="Campaign")
                                content_id = gr.Textbox(label="Content ID")
                                key_messages = gr.Textbox(label="Key Messages")
                                content_type = gr.Textbox(label="Content Type")
                                channel = gr.Textbox(label="Channel")
                                update_row_btn = gr.Button("Update Row")
                            
                            save_changes_btn = gr.Button("Save All Changes", variant="primary", visible=False)
                            with gr.Row():
                                generate_content_btn = gr.Button("Generate All Content", variant="primary", size="lg")
                                generate_changed_content_btn = gr.Button("Generate Content for Changed Items", variant="secondary")

        status_message = gr.Textbox(label="Status", interactive=False)
        current_project = gr.State("")

        def handle_generation(*inputs):
            project_name = inputs[0]
            if not project_name or project_name.strip() == "":
                return "Error: Project name cannot be empty.", None, gr.update(visible=False)
            
            status, plan_df = generate_communications_plan(*inputs)
            if plan_df is not None and not plan_df.empty:
                return status, plan_df, gr.update(visible=True)
            else:
                empty_df = pd.DataFrame(columns=['Stakeholder Profile', 'Campaign', 'Content ID', 'Key Messages', 'Content Type', 'Channel'])
                return status, empty_df, gr.update(visible=False)

        def save_plan(current_project, df):
            if not current_project:
                return "Error: Project name is empty. Please enter a valid project name.", gr.update()
            
            try:
                df = pd.DataFrame(df)  # Convert to DataFrame if it's not already
                save_plan_version(current_project, df)
                return f"Plan saved successfully for project: {current_project}", gr.update(choices=get_existing_projects())
            except Exception as e:
                logging.error(f"Error saving plan: {str(e)}")
                return f"Error saving plan: {str(e)}", gr.update()

        def load_existing_plan(selected_project):
            plan_doc = plans_collection.find_one({'project_name': selected_project})
            if plan_doc:
                if 'current_plan' in plan_doc:
                    plan_df = pd.DataFrame(plan_doc['current_plan'])
                elif 'plan' in plan_doc:
                    plan_df = pd.DataFrame(plan_doc['plan'])
                elif 'versions' in plan_doc and plan_doc['versions']:
                    plan_df = pd.DataFrame(plan_doc['versions'][-1]['plan'])
                else:
                    empty_df = pd.DataFrame(columns=['Stakeholder Profile', 'Campaign', 'Content ID', 'Key Messages', 'Content Type', 'Channel'])
                    return selected_project, "No valid plan structure found.", empty_df, gr.update(visible=False)
                
                return selected_project, "Loaded existing plan.", plan_df, gr.update(visible=True)
            else:
                empty_df = pd.DataFrame(columns=['Stakeholder Profile', 'Campaign', 'Content ID', 'Key Messages', 'Content Type', 'Channel'])
                return selected_project, "No existing plan found.", empty_df, gr.update(visible=False)

        def close_loaded_plan():
            empty_df = pd.DataFrame(columns=['Stakeholder Profile', 'Campaign', 'Content ID', 'Key Messages', 'Content Type', 'Channel'])
            return "", "Closed loaded plan.", empty_df, gr.update(visible=False)

        def refresh_projects():
            projects = get_existing_projects()
            return gr.update(choices=projects)

        def get_existing_projects():
            projects = plans_collection.distinct('project_name')
            return sorted(projects)

        
        def save_edited_plan(current_project, df):
            try:
                df = pd.DataFrame(df)  # Convert to DataFrame if it's not already
                if not current_project:
                    return "Error: No project selected. Please load a project first.", gr.update()
                
                save_plan_version(current_project, df)
                
                return f"Changes saved successfully for project: {current_project}", gr.update(choices=get_existing_projects())
            except Exception as e:
                logging.error(f"Error saving edited plan: {str(e)}")
                return f"Error saving changes: {str(e)}", gr.update()
            
        def on_generate_changed_content_click(project_name):
            if not project_name:
                return "Error: No project selected. Please select a project first."
            
            result = generate_content_for_changed_rows(project_name)
            return result
        
        save_changes_btn.click(
            save_edited_plan,
            inputs=[current_project, existing_plan_output],
            outputs=[status_message, existing_projects]
        )

        generate_btn.click(
            handle_generation,
            inputs=[project_name, change_type, description, stakeholder_profiles, channels, market, industry, supporting_docs],
            outputs=[status_message, new_project_plan_output, save_plan_btn]
        )

        save_plan_btn.click(
            save_plan,
            inputs=[project_name, new_project_plan_output],
            outputs=[status_message, existing_projects]
        )

        load_plan_btn.click(
            load_existing_plan,
            inputs=[existing_projects],
            outputs=[current_project, status_message, existing_plan_output, close_plan_btn]
        )

        close_plan_btn.click(
            close_loaded_plan,
            outputs=[current_project, status_message, existing_plan_output, close_plan_btn]
        )

        refresh_projects_btn.click(
            refresh_projects,
            outputs=[existing_projects]
        )

        generate_content_btn.click(
            generate_content_for_plan,
            inputs=[existing_plan_output, current_project, project_name],
            outputs=[status_message]
        )

        generate_changed_content_btn.click(
            on_generate_changed_content_click,
            inputs=[current_project],
            outputs=[status_message]
        )

        

        def add_stakeholder_profile(name, areas_of_interest, areas_of_concern, current_state, future_state, 
                                    key_changes, negative_impact, total_impact, interest, influence, size, internal_external, current_profiles):
            if name and key_changes and total_impact and interest and influence and size is not None and internal_external:
                if not any(p[0] == name for p in current_profiles):
                    new_profile = [name, areas_of_interest, areas_of_concern, current_state, future_state, 
                                   key_changes, negative_impact, total_impact, interest, influence, size, internal_external]
                    current_profiles.append(new_profile)
                    return (
                        current_profiles,
                        gr.update(value=current_profiles),
                        "",  # status message
                        "",  # clear profile_name
                        "",  # clear areas_of_interest
                        "",  # clear areas_of_concern
                        "",  # clear current_state
                        "",  # clear future_state
                        "",  # clear key_changes
                        "Low",  # reset negative_change_impact
                        "Low",  # reset total_change_impact
                        "Low",  # reset change_interest
                        "Low",  # reset change_influence
                        None,  # clear size
                        "Internal",  # reset internal_external
                    )
                else:
                    return (current_profiles, gr.update(value=current_profiles), "Profile name must be unique.") + ("",) * 12
            else:
                return (current_profiles, gr.update(value=current_profiles), "Please fill in all mandatory fields.") + ("",) * 12

        add_profile_btn.click(
            add_stakeholder_profile,
            inputs=[profile_name, areas_of_interest, areas_of_concern, current_state, future_state, 
                    key_changes, negative_change_impact, total_change_impact, change_interest, change_influence, size, internal_external, stakeholder_profiles],
            outputs=[stakeholder_profiles, profile_list, status_message, profile_name, areas_of_interest, areas_of_concern, current_state, future_state, 
                     key_changes, negative_change_impact, total_change_impact, change_interest, change_influence, size, internal_external]
        )

        def extract_and_add_profiles(files, current_profiles):
            if not files:
                return current_profiles, gr.update(value=current_profiles), "No files uploaded. Please upload documents first."

            extracted_profiles = extract_stakeholder_profiles(files)
            
            new_profiles_added = 0
            for profile in extracted_profiles:
                if not any(p[0] == profile[0] for p in current_profiles):
                    current_profiles.append(profile)
                    new_profiles_added += 1

            if new_profiles_added > 0:
                return current_profiles, gr.update(value=current_profiles), f"Extracted and added {new_profiles_added} new profiles."
            else:
                return current_profiles, gr.update(value=current_profiles), "No new profiles were extracted. This could be due to parsing issues or no new unique profiles found."

        extract_profiles_btn.click(
            extract_and_add_profiles,
            inputs=[supporting_docs, stakeholder_profiles],
            outputs=[stakeholder_profiles, profile_list, status_message]
        )

        def get_selected_index(data):
            if data is not None and not data.empty and data['Name'].notna().any():
                return data.index[data['Name'].notna()][0]
            return None

        def delete_selected_profile(selected_data, current_profiles):
            selected_index = get_selected_index(selected_data)
            if selected_index is None:
                return current_profiles, gr.update(value=current_profiles), "No profile selected for deletion."
            
            if 0 <= selected_index < len(current_profiles):
                deleted_profile = current_profiles.pop(selected_index)
                return current_profiles, gr.update(value=current_profiles), f"Deleted profile: {deleted_profile[0]}"
            else:
                return current_profiles, gr.update(value=current_profiles), "Invalid selection. No profile deleted."

        delete_profile_btn.click(
            delete_selected_profile,
            inputs=[profile_list, stakeholder_profiles],
            outputs=[stakeholder_profiles, profile_list, status_message]
        )

        def display_edit_form(df, evt: gr.SelectData):
            if df is not None and not df.empty:
                row = df.iloc[evt.index[0]]
                return {
                    edit_form: gr.update(visible=True),
                    row_index: evt.index[0],
                    stakeholder_profile: row["Stakeholder Profile"],
                    campaign: row["Campaign"],
                    content_id: row["Content ID"],
                    key_messages: row["Key Messages"],
                    content_type: row["Content Type"],
                    channel: row["Channel"],
                    save_changes_btn: gr.update(visible=True),
                }
            return {edit_form: gr.update(visible=False)}

        def update_plan_row(df, index, stakeholder, campaign, content_id, messages, content_type, channel):
            df = pd.DataFrame(df)  # Convert to DataFrame if it's not already
            df.iloc[int(index)] = [stakeholder, campaign, content_id, messages, content_type, channel]
            return df, gr.update(visible=False)

        existing_plan_output.select(
            display_edit_form,
            inputs=[existing_plan_output],
            outputs=[edit_form, row_index, stakeholder_profile, campaign, content_id, key_messages, content_type, channel, save_changes_btn]
        )

        update_row_btn.click(
            update_plan_row,
            inputs=[existing_plan_output, row_index, stakeholder_profile, campaign, content_id, key_messages, content_type, channel],
            outputs=[existing_plan_output, edit_form]
        )

        new_project_plan_output.select(
            lambda df, evt: display_edit_form(df, evt, is_new_project=True),
            inputs=[new_project_plan_output],
            outputs=[new_project_edit_form, new_project_row_index, new_project_stakeholder_profile, new_project_campaign, 
                     new_project_content_id, new_project_key_messages, new_project_content_type, new_project_channel, new_project_save_changes_btn]
        )

        new_project_update_row_btn.click(
            update_plan_row,
            inputs=[new_project_plan_output, new_project_row_index, new_project_stakeholder_profile, new_project_campaign, 
                    new_project_content_id, new_project_key_messages, new_project_content_type, new_project_channel],
            outputs=[new_project_plan_output, new_project_edit_form]
        )

    return page

def content_generation_page():
    with gr.Blocks() as page:
        gr.Markdown("View, edit, and customise generated content based on your communication plan.")

        with gr.Row():
            # Left column
            with gr.Column(scale=1):
                refresh_btn = gr.Button("Refresh Projects")
                project_dropdown = gr.Dropdown(choices=[], label="Select Project")
                campaign_filter = gr.Dropdown(
                    choices=["All", "Awareness", "Desire", "Knowledge", "Ability", "Reinforcement"],
                    label="Filter by Campaign",
                    value="All"
                )
                language_filter = gr.Dropdown(
                    choices=["All", "English", "Spanish", "French", "German", "Chinese", "Japanese"],
                    label="Filter by Language",
                    value="All"
                )
                content_selector = gr.Dropdown(choices=[], label="Select Content Item", multiselect=True)

            # Right column
            with gr.Column(scale=3):
                with gr.Tabs() as content_tabs:
                    with gr.TabItem("View & Manage Content"):
                        edited_content_display = gr.Textbox(label="Content", lines=20)
                        
                        with gr.Row():
                            style_input = gr.Textbox(label="Style", placeholder="e.g. formal, casual, technical")
                            keywords_input = gr.Textbox(label="Keywords/Phrases", placeholder="Comma-separated list")
                            wordcount_input = gr.Textbox(label="Target Word Count", placeholder="Enter a number")
                            perspective_input = gr.Textbox(label="Perspective", placeholder="e.g. first-person, third-person")
                            general_prompt = gr.Textbox(label="Prompt for other changes", placeholder="Enter additional instructions or context")
                        
                        with gr.Row():
                            apply_changes_btn = gr.Button("Apply Changes", variant="primary")
                            save_changes_btn = gr.Button("Save Changes", variant="primary")
                        
                        with gr.Accordion("Global Transformations", open=False):
                            with gr.Row():
                                target_language = gr.Dropdown(
                                    choices=["Spanish", "French", "German", "Chinese", "Japanese"],
                                    label="Select Country"
                                )
                                translate_and_save_btn = gr.Button("Translate and Save", variant="primary")

                status_message = gr.Textbox(label="Status", interactive=False)

        def get_project_options():
            projects = content_collection.distinct('project_name')
            return sorted(projects)

        def update_content_selector(project, campaign="All", language="All"):
            if not project:
                return gr.update(choices=[])
            query = {'project_name': project}
            if campaign != "All":
                query['campaign'] = campaign
            if language != "All":
                query['language'] = language
            content_items = list(content_collection.find(query).sort('created_at', -1))
            logging.info(f"Content items found for project {project}: {len(content_items)}")
            choices = []
            for item in content_items:
                status = item.get('status', 'draft').capitalize()
                language = item.get('language', 'English')
                display_name = f"{item.get('content_name', 'Unknown')} - {status} - {language}"
                choices.append((display_name, str(item['_id'])))
            return gr.update(choices=choices)

        def display_selected_content(content_ids):
            if not content_ids:
                return ""
            content = ""
            for content_id in content_ids:
                content_item = content_collection.find_one({'_id': ObjectId(content_id)})
                if content_item:
                    content += content_item['content'] + "\n\n---\n\n"
            return content

        def edit_content_with_parameters(content, general_prompt, style, keywords, wordcount, perspective):
            if not content:
                return content, "Please select content to edit."
            try:
                # Convert wordcount to int if possible, otherwise use None
                try:
                    wordcount = int(wordcount)
                except ValueError:
                    wordcount = None

                prompt = f"""
                Edit the following content:

                {content}

                General instructions: {general_prompt}

                Apply the following changes:
                - Style: {style}
                - Keywords/Phrases to include: {keywords}
                - Target word count: {wordcount if wordcount else 'Not specified'}
                - Perspective: {perspective}

                Please maintain the overall message and key points while applying these changes and following the general instructions.
                """
                
                edited_content = edit_content_with_ai(content, prompt)
                return edited_content, "Content edited successfully. Review the changes and save if satisfied."
            except Exception as e:
                return content, f"Error editing content: {str(e)}"

        def save_edited_content_to_mongodb(project, content_ids, edited_content):
            try:
                for content_id in content_ids:
                    original_content = content_collection.find_one({'_id': ObjectId(content_id)})
                    if not original_content:
                        return f"Original content not found for ID: {content_id}"

                    # Create a new document for the edited content
                    new_content = original_content.copy()
                    new_content.pop('_id')  # Remove the original ID
                    new_content['content'] = edited_content
                    new_content['is_edited'] = True
                    new_content['original_id'] = ObjectId(content_id)
                    new_content['edited_at'] = datetime.now()

                    # Insert the new document
                    result = content_collection.insert_one(new_content)

                    if not result.inserted_id:
                        return f"Failed to save edited content for ID: {content_id}"

                return "Edited content saved successfully for all selected items."
            except Exception as e:
                logging.error(f"Error saving edited content to MongoDB: {str(e)}")
                return f"Error saving edited content: {str(e)}"

        def translate_content(content, target_language):
            try:
                llm = ChatOpenAI(temperature=0.7)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"You are a professional translator. Translate the following content into {target_language}. Maintain the original formatting and structure as much as possible."),
                    ("human", "{content}")
                ])
                chain = prompt | llm
                result = chain.invoke({"content": content})
                return result.content
            except Exception as e:
                logging.error(f"Error translating content: {str(e)}")
                return f"Error translating content: {str(e)}"

        def save_translated_content_to_mongodb(project, content_ids, translated_content, target_language):
            try:
                for content_id in content_ids:
                    original_content = content_collection.find_one({'_id': ObjectId(content_id)})
                    if not original_content:
                        return f"Original content not found for ID: {content_id}"

                    # Create a new document for the translated content
                    new_content = original_content.copy()
                    new_content.pop('_id')  # Remove the original ID
                    new_content['content'] = translated_content
                    new_content['is_translated'] = True
                    new_content['original_id'] = ObjectId(content_id)
                    new_content['translated_at'] = datetime.now()
                    new_content['language'] = target_language
                    new_content['content_name'] = f"{new_content['content_name']} - {target_language}"

                    # Insert the new document
                    result = content_collection.insert_one(new_content)

                    if not result.inserted_id:
                        return f"Failed to save translated content for ID: {content_id}"

                return "Translated content saved successfully for all selected items."
            except Exception as e:
                logging.error(f"Error saving translated content to MongoDB: {str(e)}")
                return f"Error saving translated content: {str(e)}"

        def refresh_projects():
            projects = get_project_options()
            logging.info(f"Refreshed projects: {projects}")
            return gr.update(choices=projects)

        def log_project_selection(project):
            logging.info(f"Selected project: {project}")
            return project

        def translate_and_save(project, content_ids, content, target_language):
            try:
                translated_content = translate_content(content, target_language)
                save_result = save_translated_content_to_mongodb(project, content_ids, translated_content, target_language)
                return translated_content, save_result
            except Exception as e:
                logging.error(f"Error in translate_and_save: {str(e)}")
                return content, f"Error: {str(e)}"

        refresh_btn.click(
            refresh_projects,
            outputs=[project_dropdown]
        )

        project_dropdown.change(
            log_project_selection,
            inputs=[project_dropdown],
            outputs=[project_dropdown]
        ).then(
            update_content_selector,
            inputs=[project_dropdown, campaign_filter, language_filter],
            outputs=[content_selector]
        )

        campaign_filter.change(
            update_content_selector,
            inputs=[project_dropdown, campaign_filter, language_filter],
            outputs=[content_selector]
        )

        language_filter.change(
            update_content_selector,
            inputs=[project_dropdown, campaign_filter, language_filter],
            outputs=[content_selector]
        )

        content_selector.change(
            display_selected_content,
            inputs=[content_selector],
            outputs=[edited_content_display]
        )

        apply_changes_btn.click(
            edit_content_with_parameters,
            inputs=[edited_content_display, general_prompt, style_input, keywords_input, wordcount_input, perspective_input],
            outputs=[edited_content_display, status_message]
        )

        save_changes_btn.click(
            save_edited_content_to_mongodb,
            inputs=[project_dropdown, content_selector, edited_content_display],
            outputs=[status_message]
        ).then(
            update_content_selector,
            inputs=[project_dropdown, campaign_filter, language_filter],
            outputs=[content_selector]
        )

        translate_and_save_btn.click(
            translate_and_save,
            inputs=[project_dropdown, content_selector, edited_content_display, target_language],
            outputs=[edited_content_display, status_message]
        ).then(
            update_content_selector,
            inputs=[project_dropdown, campaign_filter, language_filter],
            outputs=[content_selector]
        )

    return page

def content_scheduler_page():
    with gr.Blocks() as page:
        gr.Markdown("View, edit, and schedule your generated change communication content.")

        with gr.Row():
            # Left column
            with gr.Column(scale=1):
                refresh_btn = gr.Button("Refresh Projects")
                project_dropdown = gr.Dropdown(choices=[], label="Select Project")
                campaign_filter = gr.Dropdown(
                    choices=["All", "Awareness", "Desire", "Knowledge", "Ability", "Reinforcement"],
                    label="Filter by Campaign",
                    value="All"
                )

                language_filter = gr.Dropdown(
                    choices=["All", "English", "Spanish", "French", "German", "Chinese", "Japanese"],
                    label="Filter by Language",
                    value="All"
                )

                status_filter = gr.Radio(["All", "Draft", "Scheduled", "Delivered"], label="Filter by Status", value="All")
                
                content_selector = gr.Dropdown(choices=[], label="Select Content Item")
                
                recipient = gr.Textbox(label="Recipient Email")
                send_date = gr.Textbox(label="Send Date (YYYY-MM-DD)")
                send_time = gr.Textbox(label="Send Time (HH:MM)")
                schedule_btn = gr.Button("Schedule Email")

            # Right column
            with gr.Column(scale=2):
                content_preview = gr.Textbox(label="Content Preview", lines=40)

        # Scheduled emails list at the bottom
        scheduled_list = gr.Dataframe(
            headers=["Project", "Content ID", "Content Description", "Recipient", "Scheduled Time", "Status"],
            label="Scheduled Emails",
            value=scheduled_emails
        )

        status_message = gr.Textbox(label="Status", interactive=False)

        def refresh_projects():
            projects = get_project_options()
            return gr.Dropdown(choices=projects)

        def update_content_selector(project, campaign="All", status="All", language="All"):
            if not project:
                return gr.update(choices=[])
            query = {'project_name': project}
            if campaign != "All":
                query['campaign'] = campaign
            if status != "All":
                query['status'] = status.lower()
            if language != "All":
                query['language'] = language
            content_items = list(content_collection.find(query).sort('created_at', -1))
            choices = []
            for item in content_items:
                status = item.get('status', 'draft').capitalize()
                language = item.get('language', 'English')
                display_name = f"{item['content_name']} - {status} - {language}"
                choices.append((display_name, str(item['_id'])))
            return gr.update(choices=choices)

        def display_selected_content(content_id):
            if not content_id:
                return ""
            content_item = content_collection.find_one({'_id': ObjectId(content_id)})
            return content_item['content'] if content_item else ""

        def schedule_email(project, content_id, content, recipient, date, time):
            global scheduled_emails
            try:
                if not all([project, content_id, recipient, date, time]):
                    return scheduled_emails, "Please fill in all fields."
                
                try:
                    datetime.strptime(time, "%H:%M")
                except ValueError:
                    return scheduled_emails, "Invalid time format. Please use HH:MM."
                
                scheduled_time = f"{date} {time}"
                
                content_item = content_collection.find_one({'_id': ObjectId(content_id)})
                if not content_item:
                    return scheduled_emails, f"Content item with ID {content_id} not found."
                
                # Use get() method with default values to avoid KeyError
                display_content = (f"{content_item.get('type', 'Unknown')} for "
                                   f"{content_item.get('audience', 'Unknown')} - "
                                   f"{content_item.get('campaign', 'Unknown')}")
                
                new_row = [project, content_id, display_content, recipient, scheduled_time, 'Scheduled']
                scheduled_emails.append(new_row)
                
                # Update content status in MongoDB
                content_collection.update_one(
                    {'_id': ObjectId(content_id)},
                    {'$set': {'status': 'scheduled'}}
                )
                
                print(f"Email scheduled: {new_row}")
                return scheduled_emails, f"Email scheduled for {scheduled_time}"
            except Exception as e:
                print(f"Error scheduling email: {e}")
                return scheduled_emails, f"Error scheduling email: {str(e)}"

        refresh_btn.click(
            refresh_projects,
            outputs=[project_dropdown]
        )

        project_dropdown.change(
            update_content_selector,
            inputs=[project_dropdown, campaign_filter, status_filter, language_filter],
            outputs=[content_selector]
        )

        campaign_filter.change(
            update_content_selector,
            inputs=[project_dropdown, campaign_filter, status_filter, language_filter],
            outputs=[content_selector]
        )

        status_filter.change(
            update_content_selector,
            inputs=[project_dropdown, campaign_filter, status_filter, language_filter],
            outputs=[content_selector]
        )

        language_filter.change(
            update_content_selector,
            inputs=[project_dropdown, campaign_filter, status_filter, language_filter],
            outputs=[content_selector]
        )

        content_selector.change(
            display_selected_content,
            inputs=[content_selector],
            outputs=[content_preview]
        )

        schedule_btn.click(
            schedule_email,
            inputs=[project_dropdown, content_selector, content_preview, recipient, send_date, send_time],
            outputs=[scheduled_list, status_message]
        )

    return page

#background image for gradio home page ta
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'image.jpg')
logo_path = os.path.join(current_dir, 'Logo.jpg')


custom_css = """
#logo {
    position: absolute;
    top: 10px;
    right: 10px;
    max-width: 68px;
    height: 68px;
    border: none;
    box-shadow: none;
    background: none;
}
.about-background {
    background-image: url('file=image.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    padding: 20px;
    border-radius: 10px;
}

.about-background .markdown-body {
    background-color: rgba(255, 255, 255, 0.8);
    padding: 20px;
    border-radius: 10px;
}
"""

# Main application
with gr.Blocks(theme=gr.themes.Soft(), fill_width=True, css=custom_css) as demo:
    with gr.Row():
        gr.Markdown("# MetamorphAI", elem_id="title")
        gr.Image(logo_path, elem_id="logo", show_label=False, width=68, height=68, show_download_button=False, interactive=False)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("About"):
            about_page()
        with gr.TabItem("Communications Strategy"):
            change_accelerator_page()
        with gr.TabItem("Content Manager"):
            content_generation_page()
        with gr.TabItem("Communications Delivery"):
            content_scheduler_page()

# Initialize email sender worker
def initialize():
    threading.Thread(target=email_sender_worker, daemon=True).start()
    update_existing_content_status()

if __name__ == "__main__":
    initialize()  # Call initialize() first
    demo.launch(allowed_paths=[current_dir])  # Then launch the Gradio interface