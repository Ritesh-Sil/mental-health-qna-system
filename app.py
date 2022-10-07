import streamlit as st
import numpy as np
import pandas as pd
import re
import joblib
from bs4 import BeautifulSoup


# -------------------- All the utility functions --------------------
def fn_preprocess_text(sentence):
    text = str(sentence).lower()

    text = text.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")
    text = text.replace("what's", "what is").replace("it's", "it is").replace("i'm", "i am")
    text = text.replace("he's", "he is").replace("she's", "she is").replace("'s", " own")
    text = text.replace("'ll", " will").replace("n't", " not").replace("'re", " are").replace("'ve", " have")
    text = text.replace("?", "").replace("i'm", " i am").replace("what's", " what is")

    text = re.sub('[^a-zA-Z0-9\n]', ' ', text)  # ------------------- Replace every special char with space
    text = re.sub('\s+', ' ', text).strip()  # ---------------------- Replace excess whitespaces

    text = BeautifulSoup(text).get_text()
    return text


def fn_term_frequency(preprocessed_text):
    # Import the Counter class
    from collections import Counter

    # Create the wordlist
    Word_list = preprocessed_text.split()

    # Find the total word count
    total_word_count = len(Word_list)

    # Find each word count
    Word_counts = dict(Counter(Word_list))

    # Calculate the term frequency for each word and return it in form of a dictionary
    term_freq = {}
    for word in Word_counts.keys():
        term_freq[word] = Word_counts[word] / total_word_count
    # print(term_freq)
    return term_freq


def fn_tfidf_input(preprocessed_text, df_idf):
    # Find the term frequency of the input text
    term_freq = fn_term_frequency(preprocessed_text)

    # Create the vector of the input text based on the available vocabulary in the dataset
    vec = []
    for words in df_idf.words.values:
        if words in term_freq.keys():
            vec.append(term_freq[words])
        else:
            vec.append(0)

    # Create the tf-idf transformation matrix and return it
    tfidf_transformed = (np.array(vec) * df_idf.idf.values).reshape(300, 1).T
    return tfidf_transformed


def fn_tfidf_input_df(df, df_idf):
    # Initializing empty dataframe
    df_q = pd.DataFrame()
    df_a = pd.DataFrame()

    # Transforming each question and answer pair into tf-idf vectors
    for i in range(len(df)):
        df_tfidf_q = pd.DataFrame(fn_tfidf_input(df['questions'][i], df_idf))
        df_q = df_q.append(df_tfidf_q)

        df_tfidf_a = pd.DataFrame(fn_tfidf_input(df['answers'][i], df_idf))
        df_a = df_a.append(df_tfidf_a)

    # Creating the final dataframe
    df_tfidf = pd.concat([df_q, df_a], axis=1)
    df_tfidf.columns = range(df_tfidf.columns.size)
    return df_tfidf


def fn_df_X_stat_feats(df):
    from fuzzywuzzy import fuzz

    # Finding the length of questions and the answers
    str_lg_q = [len(i) for i in df.questions.values]
    str_lg_a = [len(i) for i in df.answers.values]

    # Finding the number of words in questions and the answers
    n_words_q = [len(i.split()) for i in df.questions.values]
    n_words_a = [len(i.split()) for i in df.answers.values]

    # Finding the common words between questions and the answers
    common_words = [len(set(questions.split()) & set(answers.split())) for questions, answers in
                    zip(df.questions.values, df.answers.values)]

    # Finding the unique words between questions and the answers
    unique_words = [len(set(questions.split()) | set(answers.split())) for questions, answers in
                    zip(df.questions.values, df.answers.values)]

    set_ratio = [fuzz.token_set_ratio(questions, answers) for questions, answers in
                 zip(df.questions.values, df.answers.values)]
    sort_ratio = [fuzz.token_sort_ratio(questions, answers) for questions, answers in
                  zip(df.questions.values, df.answers.values)]
    fuzz_ratio = [fuzz.QRatio(questions, answers) for questions, answers in zip(df.questions.values, df.answers.values)]
    partial_ratio = [fuzz.partial_ratio(questions, answers) for questions, answers in
                     zip(df.questions.values, df.answers.values)]

    kw = dict(str_lg_q=str_lg_q, str_lg_a=str_lg_a, n_words_a=n_words_a,
              n_words_q=n_words_q, common_words=common_words, unique_words=unique_words,
              set_ratio=set_ratio, sort_ratio=sort_ratio, fuzz_ratio=fuzz_ratio,
              partial_ratio=partial_ratio)

    df_X_stat_feats = pd.DataFrame().assign(**kw)
    return df_X_stat_feats


def fn_standardized_input(df, df_mean_std):
    # Create the dataframe and store the average and standard deviation
    #df_mean_std = pd.DataFrame().assign(average=std_obj_csv.mean_, stdev=std_obj_csv.var_ ** 0.5)

    # Calculate deviation from mean
    df_deviation = df.values - df_mean_std.average.values

    # Standardize
    df_std = pd.DataFrame(df_deviation / df_mean_std.stdev.values)

    return df_std

# ----------------    Main function        --------------------


def main():
    # Title of the application
    st.title('Mental Health Question Answering System')

    # File uploads
    df_qna_csv = st.file_uploader("Choose the question answer file")
    df_idf_csv = st.file_uploader("Choose the csv file with idf info")
    df_std_csv = st.file_uploader("Choose the csv file with mean-standard deviation matrix")
    model = st.file_uploader("Upload the model")


    # User input text and preprocessing
    text = st.text_input("Please ask a question")
    text = fn_preprocess_text(text)
    st.text(text)

    # QnA pair uploader
    # df_qna_csv = st.file_uploader("Choose the question answer file")
    df_qna = pd.read_csv(df_qna_csv).iloc[:,-2:]
    df_qna['questions'] = text
    # st.dataframe(df_qna)

    # Word - Idf file upload
    # df_idf_csv = st.file_uploader("Choose the csv file with idf info")
    df_idf = pd.read_csv(df_idf_csv)

    # Tf-idf transformation
    df_tfidf_qa = fn_tfidf_input_df(df_qna, df_idf)
    df_tfidf_qa = df_tfidf_qa.reset_index()
    df_tfidf_qa = df_tfidf_qa.iloc[:,1:]
    # st.dataframe(df_tfidf_qa)

    # Fuzzy features addition
    token_set_ratio_ip = fn_df_X_stat_feats(df_qna)['set_ratio'].values.reshape(-1,1)
    token_sort_ratio_ip = fn_df_X_stat_feats(df_qna)['sort_ratio'].values.reshape(-1,1)
    df_stat_feat = pd.DataFrame(np.concatenate([token_set_ratio_ip,token_sort_ratio_ip],axis =1),columns=['set_ratio','sort_ratio'])
    df_tfidf_stat_feat = pd.concat([df_tfidf_qa,df_stat_feat],axis=1)
    # st.dataframe(df_tfidf_stat_feat)

    # Tf-idf QnA dot product addition
    tfidf_q_input = np.array(df_tfidf_stat_feat.iloc[:,:300])
    tfidf_a_input = np.array(df_tfidf_stat_feat.iloc[:,300:600])
    tdidf_dot_prod_input = [i @ j for i, j in zip(tfidf_q_input, tfidf_a_input)]
    df_all_feat = df_tfidf_stat_feat.assign(tdidf_dot_prod_input = tdidf_dot_prod_input)
    # st.dataframe(df_all_feat)

    # Standardize the dataset
    # df_std_csv = st.file_uploader("Choose the csv file with mean-standard deviation matrix")
    df_std = pd.read_csv(df_std_csv)
    df_all_feat_std = fn_standardized_input(df_all_feat, df_std)
    # st.dataframe(df_all_feat_std)

    # Load the model and predict
    # model = st.file_uploader("Upload the model")
    classifier = joblib.load(model)
    probability = classifier.predict_proba(df_all_feat_std).T[1]
    df_prob = pd.DataFrame().assign(probability = probability)
    # st.dataframe(df_prob)

    # Answers - Probability dataframe
    df_ans = df_qna['answers']
    df_ans_prob = pd.concat([df_ans,df_prob],axis=1).sort_values(by = 'probability', ascending=False)
    st.dataframe(df_ans_prob.iloc[:3,:1])


if __name__ == "__main__":
    main()

