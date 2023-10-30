from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.matutils import Sparse2Corpus
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import re
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import csv
import pandas as pd

class TMPipeline():

    """
    Pipeline for topic modeling.

    Args:
        file_path (str): The path to the file containing the documents.
        pipeline (str): The type of topic modeling pipeline to use.
            Choices are 'TTM' for traditional topic models (LDA, NMF) or 'LLM' for the LLM based pipeline.
    """
    
    def __init__(self, file_path, pipeline):
        # Initialize the class with the given file path and pipeline choice
        self.pipeline = pipeline
        self.dataset = file_path.split('\\')[-1].replace('.txt', '')

        # Check if the pipeline is 'LLM' and the file path does not contain 'ChatGPT_Tweets'
        if self.pipeline == 'LLM' and not 'ChatGPT_Tweets' in file_path:
            encoding = None
        else:
            encoding = 'utf-8'

        # Open the file and read its contents into a list of documents
        with open(file_path, 'r', encoding=encoding) as file:
            self.documents = file.read().split('\n')

        # Ignore future warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Check if the pipeline choice is valid
        if self.pipeline not in ['TTM', 'LLM']:
            raise ValueError(f"Invalid pipeline choice: {self.pipeline}")

    def preprocess(self):
        """
        Preprocesses the documents based on the pipeline specified.

        """
        # Check if the pipeline is either TTM or LLM
        if self.pipeline == 'TTM':
            self._preprocess_ttm()
        elif self.pipeline == 'LLM':
            self._preprocess_llm()

        # Print the average and total number of words per document
        total_words = sum(len(doc) for doc in self.documents)
        num_documents = len(self.documents)
        average_word_count = total_words / num_documents
        print("Average word count:", int(average_word_count))
        print("Total words:", int(total_words))

    def _preprocess_ttm(self):
        """
        Performs preprocessing for the 'TTM' pipeline.
        """
        # Set up stop words and lemmatizer
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        # Lemmatize words, remove stop words, non-alphanumeric characters, and non-ASCII characters
        self.documents = [
            [
                lemmatizer.lemmatize(word)
                for word in document.lower().split()
                if word not in stop_words and not re.search(r'\W+|\d+', word) and all(ord(char) < 128 for char in word)
            ]
            for document in self.documents
        ]

        # Filter out words with length less than 3
        self.documents = [[word for word in doc if len(word) >= 3] for doc in self.documents]

        # Generate bigrams
        bigram = Phrases(self.documents, min_count=5)
        for idx in range(len(self.documents)):
            for token in bigram[self.documents[idx]]:
                if '_' in token:
                    self.documents[idx].append(token)

    
    def _preprocess_llm(self):
        """
        Performs preprocessing for the 'LLM' pipeline.
        """

        # Split each document by comma, remove numeric words and non-ASCII characters
        self.documents = [[word.strip() for word in document.lower().split(',') if not re.search(r'\d+', word) and all(ord(char) < 128 for char in word)] for document in self.documents]

        # Replace hyphens and spaces with underscores
        self.documents = [[word.replace('-', '_').replace(' ', '_') for word in doc] for doc in self.documents]

        # Filter out words with length less than 3
        self.documents = [[word for word in doc if len(word) >= 3] for doc in self.documents]


    def vectorize(self, vectorizer):
        """
        Vectorizes the documents based on the chosen vectorizer.
    
        Args:
            vectorizer (str): The type of vectorizer to use. Valid options are 'tf' or 'tfidf'.
    
        Raises:
            ValueError: If an invalid vectorizer choice is provided.
        """
        # Make the choice of vectorizer available across the class
        self.vec_choice = vectorizer

        def vectorize_corpus(vectorizer, documents):
            corpus = [' '.join(doc) for doc in documents]
            corpus = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            return corpus, feature_names
    
        # Initialize vectorizer based on choice
        if self.vec_choice == 'tf':
            self.vectorizer = CountVectorizer()
        elif self.vec_choice == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        else:
            raise ValueError(f"Invalid vectorizer choice: {self.vec_choice}")
    
        # Always create tf corpus
        tf_vectorizer = CountVectorizer()
        self.tf_corpus, tf_features = vectorize_corpus(tf_vectorizer, self.documents)
    
        # Create corpus based on vectorizer choice
        self.corpus, self.feature_names = vectorize_corpus(self.vectorizer, self.documents)

        # Print the shape of the vectorized corpus
        print("Shape of the vectorized corpus:", self.corpus.shape)

    def _topic_diversity(self, topk):
        """
        Calculates the topic diversity metric based on the topk words in each topic.
    
        Args:
            topk (int): The number of top words to consider for each topic.
    
        Returns:
            float: The topic diversity metric.
    
        Note:
            The topic diversity metric is calculated as the ratio of unique words across all topics
            to the total number of words considered (topk * number of topics).
        """
        unique_words = set()
        for topic in self.topics:
            unique_words.update(topic[:topk])
        puw = len(unique_words) / (topk * len(self.topics))
        return puw

    
    def model_topics(self, model, num_topics):
        """
        Models topics using the specified model and number of topics.
    
        Args:
            model (str): The topic modeling algorithm to use. Valid options are 'lda' or 'nmf' for the TTM pipeline or 'kmeans' for the LLM pipeline.
            num_topics (int): The number of topics to generate.
        """
        # Make the choice of model available across the class
        self.model = model
        
        if self.pipeline == 'TTM':
            if model == 'lda':
                # Create a Latent Dirichlet Allocation topic model with the specified number of topics
                topic_model = LatentDirichletAllocation(n_components=num_topics, random_state=0)
            elif model == 'nmf':
                # Create a Non-negative Matrix Factorization topic model with the specified number of topics
                topic_model = NMF(n_components=num_topics, random_state=0)
            else:
                raise ValueError(f"Invalid model choice: {model}")

            # Measure the inference time and fit the model to the corpus
            start_time = time.time()
            topic_model.fit(self.corpus)
            self.inference_time = time.time() - start_time

            self.topics = []
            for topic_idx, topic in enumerate(topic_model.components_):
                # Get the top words for each topic
                top_words = [self.feature_names[i] for i in topic.argsort()[:-11:-1]]
                self.topics.append(top_words)


        elif self.pipeline == 'LLM':
            if model == 'kmeans':
                # Use KMeans clustering
                # Measure the inference time and fit the model to the corpus
                start_time = time.time()
                kmeans = KMeans(n_clusters=num_topics, random_state=0).fit(self.corpus)
                self.inference_time = time.time() - start_time

                # Get the order of centroids and the feature names
                order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
                terms = self.vectorizer.get_feature_names_out()

                self.topics = []
                for i in range(num_topics):
                    topic = []
                    # Get the top terms for each topic
                    for ind in order_centroids[i, :10]:
                        topic.append(terms[ind])
                    self.topics.append(topic)

            else:
                raise ValueError(f"Invalid model choice: {model}")

        # Compute coherence score
        self.dictionary = Dictionary(self.documents)
        gensim_corpus = Sparse2Corpus(self.tf_corpus, documents_columns=False)
        cm = CoherenceModel(topics=self.topics, corpus=gensim_corpus, dictionary=self.dictionary, texts=self.documents, coherence='u_mass')
        self.coherence = cm.get_coherence()

        # Compute diversity score
        self.diversity = self._topic_diversity(10)           

    def results(self, save=False):
        """
        Prints the topic coherence, topic diversity, and the top words for each topic.
        """
        # Print the topic coherence and topic diversity scores
        topic_coherence = round(self.coherence, 2)
        topic_diversity = round(self.diversity, 2)
        print(f'Topic Coherence: {topic_coherence}')
        print(f'Topic Diversity: {topic_diversity}\n')

        # Print the top words for each topic
        for i, topic in enumerate(self.topics, start=1):
            topic_words = ' '.join(topic)
            print(f"Topic {i}: {topic_words}")

        # Print the inference time
        inference_time = round(self.inference_time, 2)
        print(f'\nInference time: {inference_time} s')

        if save == True:
            # Save the top words for each topic to a file
            with open(f'Data/Raw/Topics/{self.dataset}-{self.vec_choice}-{self.model}.txt', 'w') as f:
                for topic in self.topics:
                    f.write(' '.join(topic))
                    f.write('\n')

    def optimize(self, model, start, end, step):
        """
        Computes coherence and diversity for different number of topics, and plots the results.
    
        Args:
            start (int): The starting number of topics.
            end (int): The ending number of topics.
            step (int): The step size for incrementing the number of topics.
        """

        self.scores = {'Topics': [], 'Coherence': [], 'Diversity': []}

        total_time = 0
        topic_nums = list(range(start, end+1, step))

        for num_topics in tqdm(topic_nums, desc=f"Computing coherence and diversity for different number of topics"):
            start_time = time.time()
            self.model_topics(model, num_topics)

            self.scores['Topics'].append(num_topics)
            self.scores['Coherence'].append(self.coherence)
            self.scores['Diversity'].append(self.diversity)

            elapsed_time = time.time() - start_time 
            total_time += elapsed_time 

        # Plotting
        sns.set()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # 2 rows, 1 column

        # Heading
        heading = f"Pipeline: {self.pipeline} - Vectorizer: {self.vec_choice} - Model: {model}"
        fig.suptitle(heading, fontsize=12)
    
        # Plot coherence
        sns.lineplot(x=topic_nums, y= self.scores['Coherence'], ax=ax1, color='tab:blue')
        ax1.set_ylabel('Coherence')
    
        # Plot diversity
        sns.lineplot(x=topic_nums, y=self.scores['Diversity'], ax=ax2, color='tab:red')
        ax2.set_xlabel('Number of Topics')
        ax2.set_ylabel('Diversity')
    
        fig.tight_layout()
        #plt.show()
        figure_save_path = 'Data/Plots/'
        plt.savefig(figure_save_path + f'{self.dataset}-{self.vec_choice}-{model}.png')

        # Compute the rankings
        df = pd.DataFrame(self.scores)
        coherence_sorted_df = df.sort_values(by='Coherence', ascending=False, ignore_index=True)
        diversity_sorted_df = df.sort_values(by='Diversity', ascending=False, ignore_index=True)

        rank_dictionary = {'Topics': [], 'Combined Rank': []}

        for i in range(10, 51):
            rank_dictionary['Topics'].append(i)

            coherence_rank = coherence_sorted_df[coherence_sorted_df['Topics'] == i].index[0]
            diversity_rank = diversity_sorted_df[diversity_sorted_df['Topics'] == i].index[0]

            rank_dictionary['Combined Rank'].append(coherence_rank + diversity_rank)

        rank_df = pd.DataFrame(rank_dictionary)
        best_topics = rank_df.sort_values(by='Combined Rank').iloc[0]['Topics']
        coherence_score = self.scores['Coherence'][self.scores['Topics'].index(best_topics)]
        diversity_score = self.scores['Diversity'][self.scores['Topics'].index(best_topics)]

        # Save the results to a CSV file
        with open('Data/results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.dataset, self.vec_choice, model, best_topics, round(coherence_score, 2), round(diversity_score, 2), round(total_time, 1)])
