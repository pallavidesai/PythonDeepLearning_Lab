import nltk
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

lemmatizer = WordNetLemmatizer()
with open("data.txt",encoding='utf-8') as f:
    raw = f.read()
    #Tokenization
    tokens = nltk.word_tokenize(raw)
    for word in tokens:
        print(lemmatizer.lemmatize(word))
    print('******************************************')
    text = nltk.Text(tokens)
    bigrams = nltk.bigrams(text)
    # Bigrams
    print([' '.join(grams) for grams in ngrams(tokens, 2)])
    print('******************************************')
    freq_bi = nltk.FreqDist(bigrams)
    for k, v in freq_bi.items():
        print(k, v)
    print('******************************************')
    print("Most frequent bigrams below:")
    freq_bi.most_common(5)
    freq_bi.plot(5)
    for x in freq_bi.most_common(5):
        print(x[0])
f.close()
print('******************************************')
print("Summarization below:")
with open("data.txt",encoding='utf-8') as f:
    list = ['has a', '. Python', '. It', '] .', 'and has']
    for line in f:
        for x in list:
            if x in line:
                print(line)
                break
f.close()