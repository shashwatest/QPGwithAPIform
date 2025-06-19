import os
import traceback
import time
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from src.utils import read_file, split_text_into_chunks, split_list_into_chunks
from src.rag import create_faiss_index, extract_keywords, query_faiss_index
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain 
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import numpy as np
from src.ques_gen import quiz_generation_prompt, generate_quiz_prompt, quiz_evaluation_prompt, generate_quiz_evaluation_prompt

app = Flask(__name__)
app.secret_key = 'suman_at_nelumbus'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'pptx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return generate_questions()
    return render_template('index.html')

def generate_questions():
    try:
        lecture_plan_file = request.files.get('lecture_plan_file')
       
        if lecture_plan_file:
            lecture_plan_filename = secure_filename(lecture_plan_file.filename.strip())
            if allowed_file(lecture_plan_filename):
                lecture_plan_path = os.path.join(app.config['UPLOAD_FOLDER'], lecture_plan_filename)
                lecture_plan_file.save(lecture_plan_path)
            else:
                flash('Invalid lecture plan file total_num')
                return redirect(url_for('index'))
        else:
            flash('No lecture plan file part')
            return redirect(url_for('index'))

        data_files = request.files.getlist('data_files')
        data_file_paths = []
        if data_files:
            for file in data_files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename.strip())
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    data_file_paths.append(file_path)

        gemini_api = request.form.get('gemini_api')

        if not gemini_api:
            flash('API key is required.')
            return redirect(url_for('index'))

        university_name = request.form.get('university_name')
        department = request.form.get('department')
        subject = request.form.get('subject')
        academic_year = request.form.get('academic_year')
        semester = request.form.get('semester')
        one_num = int(request.form.get('one_num'))
        short_num = int(request.form.get('short_num'))
        long_num = int(request.form.get('long_num'))
        level = request.form.get('level')
        additional_info = request.form.get('additional_info')

        #LLM initialization for ques_gen

        llm = ChatGoogleGenerativeAI(api_key=gemini_api,model="gemini-2.0-flash",temperature=0.7)

        quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

        review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="final_paper", verbose=True)

        #LLM initialized

        #Embedding model initialization for rag
        gemini_embedding_model_name = "models/text-embedding-004"
        embedding_model = GoogleGenerativeAIEmbeddings(
            task_type="retrieval_query",
            google_api_key=gemini_api,
            model=gemini_embedding_model_name
        )

        def get_embeddings(texts):
            embeddings = embedding_model.embed_documents(texts)
            embeddings = np.array(embeddings)
            return embeddings

        def get_topic_embeddings(topics):
            if any(isinstance(i, list) for i in topics):
                topics = [item for sublist in topics for item in sublist]
            topics = [str(topic) for topic in topics]
            embeddings = embedding_model.embed_documents(topics)
            embeddings = np.array(embeddings)
            return embeddings

        #embedding model initialized

        total_num = one_num + short_num + long_num

        if not subject or not level or not one_num or not short_num or not long_num or not type:
            flash('Subject, Knowledge level, total_num of questions per type are required.')
            return redirect(url_for('index'))
 
        lecture_plan_content = read_file(lecture_plan_path)
        topics = extract_keywords(lecture_plan_content)
        topics = [str(topic) for topic in topics]
        topic_length = len(topics)


        #Fixing number of questions to be generated to counter token exceeded error.
        if data_file_paths:
            if long_num > 0 and short_num > 0 and one_num > 0 and topic_length > 20:
                ques_long = 1
                ques_short = 1
                ques_one = 1
            else:
                ques_long = int(long_num/topic_length) + 1
                ques_short = int(short_num/topic_length) + 1
                ques_one = int(one_num/topic_length) + 1 
        else:
            ques_long = long_num
            ques_short = short_num
            ques_one = one_num

        ques_total = ques_long + ques_short + ques_one

        if data_file_paths:
            text = ""
            for file_path in data_file_paths:
                text += "\n\n" + read_file(file_path)
            chunks = split_text_into_chunks(text)
            embeddings = get_embeddings(chunks)
            index = create_faiss_index(embeddings)
            # this is the number of topics to be passed to the LLM at once, decrease this number for higher relevance to the provided data, increase for more speed
            topic_num = 5
            topics = split_list_into_chunks(topics, topic_num)
        else:
            topics = split_list_into_chunks(topics, 50)

        all_questions = ""

        for topic in topics:
            topic_num = len(topic)
            topic_embedding = get_topic_embeddings([topic])

            if data_file_paths:
                relevant_chunks = query_faiss_index(index, topic_embedding, chunks, top_k=3)
                context = "\n".join(relevant_chunks)
            else:
                context = "Rely on general knowledge, no specific context has been provided."

            generate_quiz_prompt(
                topic=topic,
                subject=subject,
                ques_total=ques_total,
                ques_long=ques_long,
                ques_short=ques_short,
                ques_one=ques_one,
                level=level,
                context=context,
                topic_num=topic_num
            )

            response = quiz_chain.invoke({
                "topic": topic,
                "subject": subject,
                "ques_total": ques_total,
                "ques_long": ques_long, 
                "ques_short": ques_short,
                "ques_one": ques_one,
                "level": level,
                "context": context,
                "topic_num": topic_num
            })

            if isinstance(response, dict):
                quiz = response.get("quiz", "")
                all_questions += quiz + "\n\n"

            # Adding delay to avoid rate limiting, this next line can be removed when there is no token RPM limit.
            time.sleep(12)

        if all_questions:
            generate_quiz_evaluation_prompt(
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
                department=department
            )
            final = review_chain.invoke({
                "subject": subject,
                "total_num": total_num,
                "long_num": long_num,
                "short_num": short_num,
                "one_num": one_num,
                "all_questions": all_questions,
                "university_name": university_name,
                "academic_year": academic_year,
                "semester": semester,
                "additional_info": additional_info,
                "department": department
            })
            paper = final.get("final_paper", "")

            cleanup_files([lecture_plan_path] + data_file_paths)

            return render_template('index.html', generated_paper=paper, subject=subject)
        else:
            flash("No questions were generated. Please adjust your inputs.")
            return redirect(url_for('index'))

    except Exception as e:
        traceback.print_exception(total_num(e), e, e.__traceback__)
        flash("Something went wrong. Please try again.")
        return redirect(url_for('index'))

def cleanup_files(file_paths):
    """Delete files to clean up."""
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)