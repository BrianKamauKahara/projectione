import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
##from src.mcqgenerator.MCQGenerator import generate_evaluate_chain

#loading json file
with open('Response.json','r') as file:
    RESPONSE_JSON = json.load(file)

# Creating a title for the app
st.title("MCQs Creator Application with LangChain")

# Create a form using st.form
with st.form("user_inputs"):
    # File Upload
    uploaded_file = st.file_uploader("Upload a PDF or txt file")

    # Input Fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)

    #Subject
    subject = st.text_input("Insert Subject", max_chars=20)

    #Quiz Tone
    tone = st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple")

    # Add Button
    button = st.form_submit_button("Create MCQs")

    # Check if the button is clicked and all fields have input

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)
                # Count tokens and the cost of API call
                with get_openai_callback() as cb:
                    #You need a working API
                    """ response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                        
                    )
                    """
                # st.write(response)

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                response={'text': 'The term machine learning was coined in 1959 by Arthur Samuel, an IBM employee and pioneer in the field of computer gaming and artificial intelligence.[9][10] The synonym self-teaching computers was also used in this time period.[11][12]\n\nAlthough the earliest machine learning model was introduced in the 1950s when Arthur Samuel invented a program that calculated the winning chance in checkers for each side, the history of machine learning roots back to decades of human desire and effort to study human cognitive processes.[13] In 1949, Donald Hebb, a Canadian psychologist, published a book titled The Organization of Behavior where he introduces the Hebbian theory, discussing the neural structure or synapses between the nerve cells.[14] Hebb’s model of neurons interacting with one another sets a groundwork for how AIs and machine learning algorithms work under nodes, or artificial neurons used by computers to communicate data.[13] Other researchers who have studied on human cognitive systems contributed to the modern machine learning technologies as well including logician Walter Pitts and Warren McCulloch, who proposed the early mathematical models of neural networks to come up with algorithms that mirror human thought processes.[13]\n\nBy the early 1960s an experimental "learning machine" with punched tape memory, called Cybertron, had been developed by Raytheon Company to analyze sonar signals, electrocardiograms, and speech patterns using rudimentary reinforcement learning. It was repetitively "trained" by a human operator/teacher to recognize patterns and equipped with a "goof" button to cause it to re-evaluate incorrect decisions.[15] A representative book on research into machine learning during the 1960s was Nilsson\'s book on Learning Machines, dealing mostly with machine learning for pattern classification.[16] Interest related to pattern recognition continued into the 1970s, as described by Duda and Hart in 1973.[17] In 1981 a report was given on using teaching strategies so that a neural network learns to recognize 40 characters (26 letters, 10 digits, and 4 special symbols) from a computer terminal.[18]\n\nTom M. Mitchell provided a widely quoted, more formal definition of the algorithms studied in the machine learning field: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E."[19] This definition of the tasks in which machine learning is concerned offers a fundamentally operational definition rather than defining the field in cognitive terms. This follows Alan Turing\'s proposal in his paper "Computing Machinery and Intelligence", in which the question "Can machines think?" is replaced with the question "Can machines do what we (as thinking entities) can do?".[20]\n\nModern-day machine learning has two objectives, one is to classify data based on models which have been developed, the other purpose is to make predictions for future outcomes based on these models. A hypothetical algorithm specific to classifying data may use computer vision of moles coupled with supervised learning in order to train it to classify the cancerous moles. A machine learning algorithm for stock trading may inform the trader of future potential predictions.[21]',
 'number': 5,
 'subject': 'machine learning',
 'tone': 'simple',
 'response_json': '{"1": {"mcq": "multiple choice question", "options": {"a": "choice here", "b": "choice here", "c": "choice here", "d": "choice here"}, "correct": "correct answer"}, "2": {"mcq": "multiple choice question", "options": {"a": "choice here", "b": "choice here", "c": "choice here", "d": "choice here"}, "correct": "correct answer"}, "3": {"mcq": "multiple choice question", "options": {"a": "choice here", "b": "choice here", "c": "choice here", "d": "choice here"}, "correct": "correct answer"} }',
 'quiz': '{"1": {"mcq": "Who coined the term machine learning?", "options": {"a": "Donald Hebb", "b": "Arthur Samuel", "c": "Walter Pitts", "d": "Warren McCulloch"}, "correct": "b"}, "2": {"mcq": "What was the earliest machine learning model introduced by Arthur Samuel?", "options": {"a": "Speech recognition", "b": "Image classification", "c": "Checkers game", "d": "Pattern recognition"}, "correct": "c"}, "3": {"mcq": "Which book introduced the Hebbian theory?", "options": {"a": "The Organization of Behavior", "b": "Learning Machines", "c": "Computing Machinery and Intelligence", "d": "The History of Machine Learning"}, "correct": "a"}, "4": {"mcq": "In the 1960s, a learning machine called Cybertron was developed to analyze which of the following?", "options": {"a": "Sonar signals", "b": "Speech patterns", "c": "Electrocardiograms", "d": "All of the above"}, "correct": "d"}, "5": {"mcq": "According to Tom M. Mitchell, what is the definition of machine learning?", "options": {"a": "Improving computer performance", "b": "Learning from experience to improve task performance", "c": "Analyzing cognitive processes", "d": "Developing neural networks"}, "correct": "b"} }',
 'review': 'I am a bit strange THERE IS AN ERROR HERE THAT NEEDS TO BE FIXED!!!'}
            ##st.error("Error")

            finally:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response, dict):
                    # Extract the quiz data from the response
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                            # Display the review in a text box as well
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in table data")
                    
                    else:
                        st.write(response)

