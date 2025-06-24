# agents/profile_analyzer.py
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_profile(profile_data, job_role):
    profile_text = f"""
    About: {profile_data.get('about', '')}
    Experience: {profile_data.get('experience', '')}
    Skills: {profile_data.get('skills', '')}
    """

    # Generate Job Description for target role
    llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-7b-instruct"
)
    job_prompt = f"Write an industry-standard job description for a {job_role}."
    job_description = llm.predict(job_prompt)

    # Embed both texts
    embed = OpenAIEmbeddings()
    vectors = embed.embed_documents([profile_text, job_description])
    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    score = int(similarity * 100)

    # Use PromptTemplate properly
    prompt = PromptTemplate(
        input_variables=["profile_text", "job_description"],
        template="""
        Based on this LinkedIn profile:
        {profile_text}

        And this job description:
        {job_description}

        Give specific suggestions to improve the profile to match the job role. Be concise and clear.
        """
    )
    prompt_input = prompt.format(profile_text=profile_text, job_description=job_description)
    feedback = llm.predict(prompt_input)

    return feedback.strip(), score
