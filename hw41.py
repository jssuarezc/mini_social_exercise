import pandas as pd
import sqlite3
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import Counter
import re

def main():

    # HW 4.1 content:

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

    connection = sqlite3.connect('database.sqlite')
    posts = pd.read_sql_query('''SELECT content FROM posts''', connection)
    connection.close()

    stop_words = stopwords.words('english')
    stop_words.extend(['would', 'best', 'always', 'amazing', 'bought', 'quick' 'people', 'new', 'fun', 'think', 'know', 'believe', 'many', 'thing', 'need', 'small', 'even', 'make', 'love', 'mean', 'fact', 'question', 'time', 'reason', 'also', 'could', 'true', 'well',  'life', 'said', 'year', 'going', 'good', 'really', 'much', 'want', 'back', 'look', 'article', 'host', 'university', 'reply', 'thanks', 'mail', 'post', 'please'])
    lemmatizer = WordNetLemmatizer()

    bow_list = []
    for content in posts['content'].fillna(''):
        tokens = word_tokenize(content.lower())
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]
        tokens = [t for t in tokens if len(t) > 2 and t not in stop_words]
        if tokens:
            bow_list.append(tokens)

    dictionary = Dictionary(bow_list)
    dictionary.filter_extremes(no_below=2, no_above=0.3)
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]

    best_score, best_model, best_k = -1, None, 0
    for k in range(2, 17):
        lda = LdaModel(corpus, num_topics=k, id2word=dictionary, passes=10, random_state=42)
        coherence = CoherenceModel(model=lda, texts=bow_list, dictionary=dictionary, coherence='c_v').get_coherence()
        if coherence > best_score:
            best_score, best_model, best_k = coherence, lda, k

    print(f"Best topic count: {best_k} (Coherence={best_score:.3f})")

    print("\nTop words per topic:")
    for i, topic in best_model.print_topics(num_words=5):
        print(f"Topic {i}: {topic}")

    topic_counts = [0] * best_k
    for bow in corpus:
        topic_id = max(best_model.get_document_topics(bow), key=lambda x: x[1])[0]
        topic_counts[topic_id] += 1

    print("\n Most popular topics:")
    for i, count in sorted(enumerate(topic_counts), key=lambda x: x[1], reverse=True):
        print(f"Topic {i}: {count} posts")

    #Generate labels for topics:
        topic_labels = {
        0: "Parenting / Family",
        1: "Books / Reviews",
        2: "Fashion / Giveaways",
        3: "Personal Growth / Events",
        4: "DIY / Creativity",
        5: "Daily Life Updates",
        6: "Relationships / Emotions",
        7: "Friendship / Positivity",
        8: "Work & Motivation",
        9: "Mental Health / Wellness"
    }

    print("\n Labeled Topics:")
    for topic_id, count in sorted(enumerate(topic_counts), key=lambda x: x[1], reverse=True):
        label = topic_labels.get(topic_id, "Unknown")
        print(f"{label:30s} — {count} posts")


    # HW 4.2 content, based in 4.1:
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

    conn = sqlite3.connect('database.sqlite')
    comments = pd.read_sql_query("SELECT content FROM comments", conn)
    conn.close()

    posts['compound'] = posts['content'].fillna('').apply(lambda x: sia.polarity_scores(str(x))['compound'])
    comments['compound'] = comments['content'].fillna('').apply(lambda x: sia.polarity_scores(str(x))['compound']) if not comments.empty else pd.Series(dtype=float)

    # Classify sentiment
    def classify_sentiment(score):
        if score > 0.05:
            return 'positive'
        elif score < -0.05:
            return 'negative'
        else:
            return 'neutral'

    posts['sentiment'] = posts['compound'].apply(classify_sentiment)
    comments['sentiment'] = comments['compound'].apply(classify_sentiment) if not comments.empty else pd.Series(dtype=object)

    # --- Platform-level summary ---
    post_mean = posts['compound'].mean()
    comment_mean = comments['compound'].mean() if not comments.empty else 0

    def tone_label(score):
        if score > 0.05:
            return "POSITIVE"
        elif score < -0.05:
            return "NEGATIVE"
        return "NEUTRAL"

    print("\nPlatform Sentiment Overview:")
    print(f"Posts → Mean sentiment = {post_mean:.3f} → Overall tone: {tone_label(post_mean)}")
    if not comments.empty:
        print(f"Comments → Mean sentiment = {comment_mean:.3f} → Overall tone: {tone_label(comment_mean)}")
    else:
        print("No comments found in the database.")

    dominant_topics = []
    for bow in corpus:
        td = best_model.get_document_topics(bow)
        dominant_topics.append(max(td, key=lambda x: x[1])[0] if td else None)
    posts['dominant_topic'] = dominant_topics[:len(posts)]

    topic_summary = []
    for topic_id in range(best_k):
        topic_posts = posts[posts['dominant_topic'] == topic_id]
        if len(topic_posts) == 0:
            continue

        mean_score = topic_posts['compound'].mean()
        counts = topic_posts['sentiment'].value_counts().to_dict()

        topic_summary.append({
            'topic_id': topic_id,
            'count': len(topic_posts),
            'avg_sentiment': mean_score,
            'pos_pct': counts.get('positive', 0) / len(topic_posts) * 100,
            'neg_pct': counts.get('negative', 0) / len(topic_posts) * 100,
            'neu_pct': counts.get('neutral', 0) / len(topic_posts) * 100,
            'keywords': ', '.join([w for w, _ in best_model.show_topic(topic_id, topn=5)])
        })

    topic_summary = sorted(topic_summary, key=lambda x: x['count'], reverse=True)

    print("\nSentiment by Topic (Top 10):")
    print("Topic | Posts | Avg | +% | -% | Neutral | Keywords")
    for t in topic_summary[:10]:
        print(f"{t['topic_id']:>4} | {t['count']:>5} | {t['avg_sentiment']:>5.2f} | "
              f"{t['pos_pct']:>4.1f}% | {t['neg_pct']:>4.1f}% | {t['neu_pct']:>7.1f}% | {t['keywords']}")

if __name__ == "__main__":
    main()