from openai import OpenAI
import streamlit as st
import numpy as np
import json
from dotenv import load_dotenv
import os
from api import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY())

MEMBER_SUMMARIZER_PROMPT = "You are a chatbot that summarizes information about specific family members. As input, information about specific family members will be presented to you. Include all relevant information and summarize the input into 2 sentences. Be sure to include all user input. Although input will be in Korean, your answer should be in **English**."

INFORMATION_EXTRACTOR_PROMPT = """You will need to help classify the family into which type they fall based on information about family members and their responses to the family questionnaire. I would like to divide families into the follwing five types:
###
TYPE1
Adventurers Family:
This family type is passionate about nature and outdoor activities and is open to new experiences. Members of the Adventurers family love exploring, engaging in physical activities, and seeking adventures that bring them closer to nature.

TYPE2
Creatives Family:
The Creatives family type has a strong interest in artistic expression and creative activities. These families enjoy arts and crafts, music, and DIY projects that allow them to express their creativity and work together on aesthetically pleasing projects.

TYPE3
Home-Centered Family:
Home-Centered families prefer spending quality time at home. They enjoy activities that can be done together in the comfort of their own home, focusing on bonding and creating memories without the need to venture outside.

TYPE4
Scholars Family:
The Scholars family is deeply interested in learning and expanding their knowledge. These families value educational activities that stimulate intellectual growth and promote a love of learning among all family members.

TYPE5
Active & Healthy Family:
Families in the Active & Healthy category prioritize maintaining health and fitness and have an interest in sports. These families engage in activities that promote physical health and well-being, ensuring that all family members stay active and healthy together.
###
Extract the information corresponding to each family category from the input provided. When extracting information, be careful not to focus on certain category. Then, using the extracted information, create a summarization of each family's characteristics. Your summarization should include all the extracted information and be presented in the form of a JSON object with the following format:
{'Extracted_info': 'Information extracted from the input.', 'Summarization': 'Summarization of the input based on the extracted information.'}

Please ensure that your summarization accurately captures the key characteristics of each family type based on the extracted information, providing a clear and comprehensive overview of each family category. Your response should be flexible enough to allow for various relevant and creative summarizations. Also your response should **strictly** follow the JSON format I instructed. 
"""

FAMILY_SUMMARIZER_PROMPT = "You are a chatbot that summarizes information about specific family. As input, information about specific family will be presented to you. Include all relevant information and summarize the input into 2 sentences. Be sure to include all user input. Although input will be in Korean, your answer should be in **English**."

MISSION_GENERATOR_PROMPT = """
As a family mission chatbot, your primary task is to generate two engaging and achievable missions tailored to the unique qualities and interests of each family member. You will be provided with a summary of each family member's details, information about the family type, characteristics associated with that family type, and additional requests from the family. The missions should be designed for family participation and should be completable within 30 minutes. Emphasize collaboration and interaction among family members to foster a sense of unity and fun within the family dynamic. Your response should be flexible to accommodate diverse family types and characteristics, aligning with their specific needs and preferences. Your answer should be **in Korean**.
"""

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def cossim(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0
    else:
        similarity = dot_product / (norm_a * norm_b)
        return similarity

if 'initialize' not in st.session_state:
    st.session_state.initialize = False

if 'type_vector' not in st.session_state:
    st.session_state.type_vector = []
    
if 'family_type' not in st.session_state:
    st.session_state.family_type = []

option = st.sidebar.selectbox(
    "어떤 기능을 이용하시겠습니까?",
    ['가족 유형 검사', '가족 미션 추출']
)

def initialize():
    st.session_state.initialize = True

if option == '가족 유형 검사':

    button = st.button("분석 시작", on_click=initialize)
    if button:
        json_family_type = [
            {
                "Document": "Adventurers Family Type",
                "Family Type Description": """
                - Strong affinity for immersive nature experiences, preferring to spend their leisure time in the great outdoors.
                - Enjoys physically engaging activities such as hiking, kayaking, and mountain biking, which are not just hobbies but integral parts of their lifestyle.
                - Seeks out new adventures, often planning trips that involve exploring unfamiliar terrains, wildlife spotting, and learning survival skills.
                - Values the educational aspects of travel and nature, teaching children about environmental conservation and biodiversity through hands-on experiences.""",
                "Mission Examples": [
                    "Visit a national park to explore its vast landscapes and learn about wildlife.",
                    "Participate in a family survival challenge that involves tasks like building a shelter, navigating with a map and compass, and identifying edible plants.",
                    "Go on a bicycle tour that includes planned routes through scenic areas, combining physical activity with sightseeing."
                ]
            },
            {
                "Document": "Creatives Family Type",
                "Family Type Description": """
                - Thrives on artistic expression and creativity, frequently engaging in painting, sculpture, and other visual arts as a family.
                - Incorporates music deeply into daily life, with family members playing instruments, attending concerts, or exploring various music genres together.
                - Enjoys DIY projects that transform their living spaces, such as home renovations or garden landscaping, reflecting their creative visions.
                - Organizes regular arts and crafts sessions that encourage each family member to showcase their artistic talents, often blending education with creativity through these activities.""",
                "Mission Examples": [
                    "Organize a family painting contest where each member creates artwork based on a common theme.",
                    "Engage in a DIY home decor project, such as redesigning a room or creating handmade decorations.",
                    "Start a family band where each member learns to play an instrument or contributes vocally to create music together."
                ]
            },
            {
                "Document": "Home-Centered Family Type",
                "Family Type Description": """
                - Prefers the comfort and familiarity of home over external activities, creating a cozy and welcoming environment.
                - Engages in indoor games, movie nights, and storytelling, making their home a hub of family entertainment and relaxation.
                - Places a strong emphasis on cooking and baking together, using these activities to pass down family recipes and bond over shared meals.
                - Views their home as a sanctuary for deep conversations, personal growth, and nurturing relationships, often decorating and arranging the living space to support these goals.""",
                "Mission Examples": [
                    "Host a family cook-off where each member prepares a dish, and everyone gets to taste and rate the dishes.",
                    "Create an indoor mystery game where family members solve puzzles and clues to 'escape' a room or find a hidden treasure.",
                    "Put together a family photo album that captures memorable events and everyday moments, creating a lasting keepsake."
                ]
            },
            {
                "Document": "Scholars Family Type",
                "Family Type Description": """
                - Values intellectual development and academic pursuits, with a home environment that includes a well-stocked library and dedicated study areas.
                - Participates in educational workshops, museum visits, and science fairs, making learning a fun and regular family outing.
                - Encourages debate and discussion on a wide range of topics, fostering critical thinking and a love of knowledge in all family members.
                - Supports continuous education through online courses, tutoring sessions, and a routine that prioritizes study time and intellectual engagement.""",
                "Mission Examples": [
                    "Conduct experiments using a science kit, exploring different scientific principles and recording the results.",
                    "Challenge the family to a library book loan competition, where each member tries to read and review the most books within a month.",
                    "Visit historical cities and sites to learn about the history and heritage firsthand, making educational travel part of their routine."
                ]
            },
            {
                "Document": "Active & Healthy Family Type",
                "Family Type Description": """
                - Emphasizes a lifestyle centered around health, fitness, and active living, with scheduled times for family workouts and outdoor sports.
                - Regularly participates in local and community sports events like marathons, cycling races, and fitness challenges.
                - Maintains a diet focused on nutrition and wellness, often preparing meals together that are healthy and energizing.
                - Uses physical activity not only as a form of exercise but also as an opportunity for teaching life skills such as teamwork, discipline, and persistence.""",
                "Mission Examples": [
                    "Start a family fitness challenge that includes daily workouts and tracks progress over a period.",
                    "Organize a weekend sports tournament where family members compete in various sports like soccer, basketball, or swimming.",
                    "Embark on a healthy eating challenge where the family focuses on creating and maintaining a nutritious diet for a month."
                ]
            }
        ]

        type_description_list = []
        for i in range(len(json_family_type)):
            type_description_list.append(json_family_type[i]['Family Type Description'].lower())
            
        for i in range(len(json_family_type)):
            st.session_state.type_vector.append({})
            st.session_state.type_vector[i]['family type'] = json_family_type[i]['Document']
            st.session_state.type_vector[i]['type description'] = json_family_type[i]['Family Type Description'].strip()
            st.session_state.type_vector[i]['embedded vector'] = get_embedding(json_family_type[i]['Family Type Description'])
            
    if st.session_state.initialize:

        num_members = st.slider(
            "가족이 몇명으로 구성되나요?",
            min_value=1,
            max_value=7,
            value=3,
            step=1,
        )

        if "members_count" not in st.session_state:
            st.session_state.members_count = 0
        if "member_info" not in st.session_state:
            st.session_state.member_info = []
        if "extracted_family_info" not in st.session_state:
            st.session_state.extracted_family_info = []

        with st.form(key="Dad_information"):
            Dad_name = st.text_input("아빠의 이름은 무엇인가요?")
            Dad_age = st.number_input("아빠의 나이가 어떻게 되나요?")
            Dad_hobby = st.text_input("아빠의 취미는 무엇인가요?")
            Dad_job = st.text_input("아빠의 직업이 어떻게 되시나요?")
            Dad_weekend = st.text_input("아빠는 주말에 주로 뭘 하나요?")
            submit_button = st.form_submit_button(label='제출')

        if submit_button:
            dad_information = f"""
            Dad's name : {Dad_name},
            Dad's age : {Dad_age},
            Dad's hobby : {Dad_hobby},
            Dad's job : {Dad_job},
            Dad's weekend : {Dad_weekend}
            """

            summary = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[
                    {'role': 'system', 'content': MEMBER_SUMMARIZER_PROMPT},
                    {'role': 'user', 'content': dad_information}
                ]
            ).choices[0].message.content
            
            st.session_state.member_info.append(summary)

        with st.form(key="mom_information"):
            mom_name = st.text_input("엄마의 이름은 무엇인가요?")
            mom_age = st.number_input("엄마의 나이가 어떻게 되나요?")
            mom_hobby = st.text_input("엄마의 취미는 무엇인가요?")
            mom_job = st.text_input("엄마의 직업이 어떻게 되시나요?")
            mom_weekend = st.text_input("엄마는 주말에 주로 뭘 하나요?")
            submit_button = st.form_submit_button(label='제출')

        if submit_button:
            mom_information = f"""
            Mom's name : {mom_name},
            Mom's age : {mom_age},
            Mom's hobby : {mom_hobby},
            Mom's job : {mom_job},
            Mom's weekend : {mom_weekend}
            """

            summary = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[
                    {'role': 'system', 'content': MEMBER_SUMMARIZER_PROMPT},
                    {'role': 'user', 'content': mom_information}
                ]
            ).choices[0].message.content
            
            st.session_state.member_info.append(summary)

        with st.form(key="child1_information"):
            child_name = st.text_input("자식의 이름은 무엇인가요?")
            child_sex = st.text_input("자식의 성별은 무엇인가요?")
            child_age = st.number_input("자식의 나이가 어떻게 되나요?")
            child_hobby = st.text_input("자식의 취미는 무엇인가요?")
            child_job = st.text_input("자식의 직업이 어떻게 되시나요?")
            child_weekend = st.text_input("자식은 주말에 주로 뭘 하나요?")
            submit_button = st.form_submit_button(label='제출')

        if submit_button:
            child_information = f"""
            child's name : {child_name},
            child's age : {child_age},
            child's hobby : {child_hobby},
            child's job : {child_job},
            child's weekend : {child_weekend}
            """

            summary = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[
                    {'role': 'system', 'content': MEMBER_SUMMARIZER_PROMPT},
                    {'role': 'user', 'content': child_information}
                ]
            ).choices[0].message.content
            
            st.session_state.member_info.append(summary)
            
        with st.form(key="child2_information"):
            child_name = st.text_input("자식의 이름은 무엇인가요?")
            child_sex = st.text_input("자식의 성별은 무엇인가요?")
            child_age = st.number_input("자식의 나이가 어떻게 되나요?")
            child_hobby = st.text_input("자식의 취미는 무엇인가요?")
            child_job = st.text_input("자식의 직업이 어떻게 되시나요?")
            child_weekend = st.text_input("자식은 주말에 주로 뭘 하나요?")
            submit_button = st.form_submit_button(label='제출')

        if submit_button:
            child_information = f"""
            child's name : {child_name},
            child's age : {child_age},
            child's hobby : {child_hobby},
            child's job : {child_job},
            child's weekend : {child_weekend}
            """

            summary = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[
                    {'role': 'system', 'content': MEMBER_SUMMARIZER_PROMPT},
                    {'role': 'user', 'content': child_information}
                ]
            ).choices[0].message.content
            
            st.session_state.member_info.append(summary)

        with st.form(key='Family information'):
            important_family_activity = st.text_input("가족이 함께하는 활동에서 가장 중요하게 생각하는 것은 무엇인가요?")
            time_spent_together = st.text_input("1주일에 가족끼리 함께 보내는 시간이 얼마 정도 되나요?")
            desired_activity = st.text_input("가족끼리 새롭게 하고 싶은 활동은 무엇인가요?")
            submit_button = st.form_submit_button(label='제출')

        if submit_button:
            family_information = f"""
            Important family activity : {important_family_activity},
            Time spent together : {time_spent_together},
            Desired activity : {desired_activity}
            """
            summary = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[
                    {'role': 'system', 'content': FAMILY_SUMMARIZER_PROMPT},
                    {'role': 'user', 'content': family_information}
                ]
            ).choices[0].message.content
            
            st.session_state.member_info.append(summary)

        start_analyzing = st.button("가족 유형 확인")

        if start_analyzing:
            info_convolution = ""
            for message in st.session_state.member_info:
                info_convolution += message + "\n"
                extracted_summarized_info = client.chat.completions.create(
                    model='gpt-4-turbo-preview',
                    messages=[
                        {'role': 'system', 'content': INFORMATION_EXTRACTOR_PROMPT},
                        {'role': 'user', 'content': info_convolution}
                    ],  
                response_format={'type': 'json_object'}
                ).choices[0].message.content

            extracted_summarized_info = json.loads(extracted_summarized_info)
            family_vector = get_embedding(extracted_summarized_info['Summarization'])
            
            max_similarity = 0
            best_vector = None

            for vector in st.session_state.type_vector:
                similarity = 100 * cossim(vector['embedded vector'], family_vector).round(4)
                
                st.markdown(f"{vector['family type']} 점수 : {similarity}")
                # 지금 데모에서는 출력값을 코사인 유사도를 기준으로 하니까 너무 값이 다 높아요.. 그래서 그냥 logprob으로 구조를 갈아야할 것 같은데 지금 당장 하기엔 시간이 너무 오래 걸릴 것 같아서 일단 이대로 했습니다..!!
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_vector = vector

            if best_vector:
                st.markdown(f"가장 높은 유사도를 가진 가족 유형: {best_vector['family type']} 점수 : {max_similarity}")
                
                st.session_state.family_type.append(best_vector)

if option == "가족 미션 추출":
    family_type = st.session_state.family_type[0]
    st.header(f"FAMILY TYPE: {family_type['family type']}")
    
    Additional_needs = st.chat_input('미션 생성에서 추가적으로 고려됐으면 하는 사항이 무엇인가요?')
    
    family_info_for_mission = ""
    for summary in st.session_state.member_info:
        family_info_for_mission += summary
    family_info_for_mission += f"""가족 유형 : {family_type['family type']}, 해당 유형에 대한 설명: {family_type['type description']}"""
    
    if Additional_needs:
        family_info_for_mission += Additional_needs
        mission = client.chat.completions.create(
            model='gpt-4-turbo-preview',
            messages=[
                {'role': 'system', 'content': MISSION_GENERATOR_PROMPT},
                {'role': 'user', 'content': family_info_for_mission}
            ]
        ).choices[0].message.content
        
        with st.chat_message('assistant'):
            st.markdown(mission)