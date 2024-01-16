if __name__ == '__main__':
    from sklearn.naive_bayes import GaussianNB
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    import data
    from nltk.corpus import words
    from nltk.corpus import stopwords
    import nltk
    from nltk.tokenize import word_tokenize
    from sklearn import metrics

    # nltk.download('punkt')
    # nltk.download('stopwords')
    set_words = set(words.words())
    stop_words = set(stopwords.words('english'))
    train_data, test_data = data.load()

    def build_vocab(data):
        vocabulary = set()
        for text in data['spam']:
            for word in word_tokenize(text):
                if word in set_words:
                    vocabulary.add(word)
        for text in data['ham']:
            for word in word_tokenize(text):
                if word in set_words:
                    vocabulary.add(word)
        return vocabulary

    vocabulary = build_vocab(train_data)

    vectorizer = CountVectorizer(binary=True, vocabulary=vocabulary, stop_words=list(stop_words))

    X_train = vectorizer.fit_transform(train_data['spam'] + train_data['ham'])
    y_train = np.concatenate((np.ones(len(train_data['spam'])),
                              np.zeros(len(train_data['ham']))))
    X_test = vectorizer.transform(test_data['spam'] + test_data['ham'] )
    y_test = np.concatenate((np.ones(len(test_data['spam'])),
                              np.zeros(len(test_data['ham']))))

    model = GaussianNB()
    model.fit(X_train.toarray(), y_train)

    y_pred = model.predict(X_test.toarray())

    # Evaluate the model
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")