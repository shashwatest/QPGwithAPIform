from langchain_core.prompts import PromptTemplate

TEMPLATE = """
You are an AI assistant tasked with generating exam questions for the subject {subject} from the given topics.

Topics : {topic}

**Reasoning**:
- Think about the key concepts associated with the topics.
- Determine what aspects are important for assessment.

**Action**:
- Use the provided context to inform question creation.
- If context is not provided or is insufficient, rely on general knowledge of the subject: {subject}.

**Context**:
{context}

**Instructions**:
- Generate a set of {ques_total} questions based on the topics provied ensuring the following distribution:
    -{ques_long} long descriptive answer type questions
    -{ques_short} short-answer type questions
    -{ques_one} one-word answers type questions
- Assume attempter to have {level} level knowledge of the subject. 
- Depending on the context provided, questions can be numerical, definition-based, true-false, fill in the blanks, application-based, descriptive or multiple-choice.
- Prefer variety in the type of questions generated.
- Ensure the format of questions generated to be:
    -[Question]
    -do not add any numeration or detail or any other information you have not been asked to.
- Generate only the question, do not generate answers or any unwanted instructions.
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["topic", "subject", "level", "context", "ques_total", "topic_num", "type", "ques_long", "ques_short", "ques_one"],
    template=TEMPLATE
    )


def generate_quiz_prompt(topic, subject, level, ques_total, topic_num, context="", ques_long=1, ques_short=1, ques_one=0):
    return TEMPLATE.format(
        topic=topic,
        subject=subject,
        level=level,
        context = context,
        ques_total=ques_total,
        ques_long=ques_long,
        ques_short=ques_short,
        ques_one= ques_one,
        topic_num = topic_num
    )

TEMPLATE2 = """
Objective: Generate a question paper from a set of questions.

Role: You are an intelligent assistant tasked with creating a question paper.

Instructions:

1. **Review Questions:** I will provide you with an unorganized list of one-word anwer, short-answer and long descriptive type questions related to {subject}.
2. **Select Questions:**  From this list, select exactly {total_num} questions ensuring the following distribution:
    -{long_num} long descriptive answer type questions
    -{short_num} short-answer type questions
    -{one_num} one-word answers type questions

 While selecting questions for the question paper, consider factors such as:
    - Relevance to {subject}
    - Clarity and quality of the questions
    - Variety of topics covered

**Information to Include in the Question Paper:**

- University: {university_name}
- Department: {department}
- Subject: {subject}
- Academic Year: {academic_year}
- Semester: {semester}
- Total Marks: [To be later added by the user]
- Time Allowed: [To be later added by the user]
- General Instructions: {additional_info}. Also include any general instruction you deem suitable.


**Questions (to choose from):**
{all_questions}

**Instructions for Output Format:** 
- Include the information asked above to include in the question with proper formatting.
- Numerate the questions selected and ensure no unnecessary information is mentioned that was not asked to be included in the question paper.
- Use latex wherever necessary and strictly ensure the final question paper should be in json format with strictly not including any redundant code or comment.

Please create the question paper by formattting them as described and reviewing them to be meeting the asks.
"""

quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "number", "all_questions", "university_name"], template=TEMPLATE2)

def generate_quiz_evaluation_prompt(subject, all_questions, department, total_num, university_name="Not provided", academic_year="Not provided", semester="Not provided", additional_info="", long_num=0, short_num=0, one_num=0):
    return TEMPLATE2.format(
        subject=subject,
        total_num=total_num,
        long_num=long_num,
        short_num=short_num,
        one_num=one_num,
        all_questions=all_questions,
        university_name=university_name,
        academic_year=academic_year,
        semester=semester,
        additional_info=additional_info,
        department = department
    )