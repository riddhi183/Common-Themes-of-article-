import sys
import glob
import os
import pandas as pd
import numpy as np 
import argparse

import spacy
from spacy import displacy
import en_core_web_sm

import re
import nltk 
import string
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer 


from collections import Counter

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')



def import_files(path,list_of_articles_dir):
    '''From the user we take the directories where the articles are stored'''
    '''Note: the path should be specified where the different articles sections are stored in folders '''
    '''The list_of_articles_dir will contain the specfic section from where each article is to be imported '''
    corpus = []

    #importing the articles  
    for directory in list_of_articles_dir:
        
        file_list = glob.glob(os.path.join(os.getcwd(), '{path_given}/{name}'.format(path_given= path, name = directory), "*.txt"))

        for file_path in file_list:
            with open(file_path) as f_input:
                corpus.append(f_input.read())
    return corpus


def extract_name_entitiy_tags(articles):
    
    ''' Function used to extract named entities in the given article.Spacy module is used for 
        named entitiy recognition. 
        We will only extract the location,country, state, cities mentioned in the articles'''

    nlp = en_core_web_sm.load()

    #find out G20 countries 
    entity_tags_for_articles = []
    for article in articles:
        doc = nlp(article)
        entity_tags_for_articles.append(set([(X.text) for X in doc.ents if X.label_ == "GPE"]))
        
    return entity_tags_for_articles 

#number of articles about G20 
def import_G20_file(file_path_name):
    '''This function imports a csv file which contains the list of G20 countries 
        and some of the major cities, states in the country'''
    '''File should be a csv file'''
    '''Specify the path file'''
    '''This contains minimal information about G20 countries, it could be improvised by adding more content to it '''
    
    file_name = file_path_name + '.csv' 
    
    data = pd.read_csv(file_name)
    
    #fill the not available values with 0
    data = data.fillna(0)
    
    #dictionary is formed to keep track of which city,state belongs to a particular country 
    g20 = {}
    for (columnName, columnData) in data.iteritems():
        for val in columnData.values:
            if val != 0:
                    g20[val] = columnName
                    
    
    return g20


def num_of_articles_about_G20_countries(g20_info, location_tags_of_articles ):
    '''This functions returns the count of number of articles about the G20 countries'''
    
    count = 0
    for places in location_tags_of_articles: 
        for val in places:
            if val in g20_info:
                count += 1 
                break
    return count


def num_of_articles_of_particular_country(country,location_tags_of_articles):
    
    '''This functions returns the count the number of articles that talk the particular G20 country'''
    
    count= 0
    for places in location_tags_of_articles: 
        if country in places:
            count += 1
    return count         


def articles_with_overlapping_countries(g20,location_tags_of_articles) :
    ''' Function returns the count with the articles talking about various countries and not just one'''
    overlap = []
    for x in range(len(location_tags_of_articles)): 
        article = []
        for val in location_tags_of_articles[x]:
            if val in g20:
                article = article + [g20[val]] 
        overlap.append(set(article))

    count_overlap = 0
    for articles in overlap: 
        if len(articles) > 1: 
            count_overlap +=1 
    
    return count_overlap

def preprocess_articles(articles,add_more_stopwords):
       
        '''Different text preprocessing steps are perfomed to obtain rich information from the text'''
        
        def wordnet_pos(pos_tag):
            
            '''Tags for the words in articles '''
            '''Used for lemmatizer '''
            if pos_tag.startswith('J'):
                return wordnet.ADJ
            elif pos_tag.startswith('V'):
                return wordnet.VERB
            elif pos_tag.startswith('N'):
                return wordnet.NOUN
            elif pos_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        porter = PorterStemmer()
        
        
        stop = stopwords.words('english')
        preprocessed =[]
        
        for article in articles:
        
            text = article.lower()

            # Remove anything which is not a digit or letter except hypen
            text = re.sub('[^a-zA-Z0-9|^\-]', ' ', text)
            #print(text)

            # Remove words with digits
            text = re.sub("\S*\d\S*", "", text).strip()

            # Remove empty hyphens
            text = re.sub(' - ', ' ', text)

            #split the sentence into words 
            text = [word.strip(string.punctuation) for word in text.split(" ")]

            #remove the empty strings 
            text = [t for t in text if len(t) > 0]

            #give parts of speech tag to every word
            pos_tags = pos_tag(text)

            #lemmatizer(grouping together the different forms of a word so there could be analyzed as a single item)
            text = [WordNetLemmatizer().lemmatize(t[0], wordnet_pos(t[1])) for t in pos_tags]

            #list of stopwords plus few words which would lose the theme of articles. 
            #frequently used words which dont give us much meaning 
            stop = stop + ['say','second','sport','two','one' ,'three' ,'year', 'month', 'week','six', 'seven', 'eight','nine'] 
            
            if add_more_stopwords:
                stop = stop + add_more_stopwords

            #remove the stopwords 
            text = [x for x in text if x not in stop]

            #Stemming(gives the root/base word) 
            text = [porter.stem(word) for word in text]

            #joining back the tokens 
            text = " ".join(text)
            
            #Final preprocessed text
            preprocessed.append(text)
            
        return preprocessed


def common_theme_in_article_sections(article, add_more_stopwords):
    
    '''This function returns the common theme of the article using the bigram method'''
    
    #preprocess the text
    preprocessed = preprocess_articles(article,add_more_stopwords)
    
    #to findout the bigrams in articles
    vectorizer = CountVectorizer(ngram_range=(2,2))
    analyzer = vectorizer.build_analyzer()
    
    bigrams= []
    
    for arti in preprocessed:
        bigrams.append(analyzer(arti))
        
    theme_of_article = []
    
    #finding out the common theme of every article by finding the top 10 bigrams 
    for articles in bigrams:
        theme_of_article.append(Counter(articles).most_common(10))
    
    #creating a list of potential themes of individual article
    potential_theme_in_article_section = []
    for themes in theme_of_article:
        for val in themes:
            potential_theme_in_article_section.append(val[0])
    
    x = Counter(potential_theme_in_article_section).most_common(5)
    
    return x



if __name__ == "__main__":
    # Parsing required and optional arguments

    # directories from where articles from various sections will be imported 
    ######## takes as arg from command line, change the path as well ############

    #list_of_articles_dir = ['sport', 'business', 'entertainment', 'tech', 'politics']
    #path = 'bbc'
#store all the articles in a list. Each corresponds to a article 


    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', required=True, help='<Required> path to folder containing articles sections')
    parser.add_argument('-l','--list',  nargs='+', help='<Required> List of article sections', required=True)
    parser.add_argument('--g20', dest='g20', required=False, help='<Required> path to folder containing articles sections',default='g20')
    parser.add_argument('-ls','--stop',  nargs='*', help='<Required> List of article sections', required=False)
    parsed_args = parser.parse_args();

    path = parsed_args.path 
    list_of_articles_dir = parsed_args.list
    g20 = parsed_args.g20
    add_more_stopwords = parsed_args.stop
    #stores all the articles
    corpus = import_files(path,list_of_articles_dir)
    print('The total number of articles imported: ' + str(len(corpus)))

    #stores the location tags for the articles 
    location_tags_for_articles = extract_name_entitiy_tags(corpus)

    #dictionary of G20 countries 
    g20_info = import_G20_file(g20)

    #number of articles about G20 countries: 
    num_of_G20_articles = num_of_articles_about_G20_countries(g20_info,location_tags_for_articles)
    print("The number of articles about G20 countries are: " + str(num_of_G20_articles) + '\n' )

    countries = ['US', 'India','Japan','Turkey','EU', 'Argentina', 'Australia',	'Brazil', 'Canada', 'China','France','Germany', 'Indonesia', 'Italy', 'Mexico', 'Korea' ,'Russia', 'Saudi Arabia','South Africa' ,'the United Kingdom']
    
    for country in countries:
        print('Number of articles for {name} country'.format(name=country) ) 
        #num of articles of a particular country:
        num_of_articles_of_country = num_of_articles_of_particular_country(country,location_tags_for_articles)
        print(num_of_articles_of_country)
        
    #num of articles with overlap
    num_of_articles_with_overlap = articles_with_overlapping_countries(g20_info,location_tags_for_articles)
    print("The number of articles with overlapping countries are : " + str(num_of_articles_with_overlap)  )

    #common theme of the articles
    for section in list_of_articles_dir:
        #importing articles of a particular section
        
        articles_in_section=import_files(path,[section])
        
        #to find the common theme in that section 
        common_theme = common_theme_in_article_sections(articles_in_section,add_more_stopwords) 
        print('The common theme for the article section: ' + str(section) )
        
        for x in common_theme:
            print(x[0])
        print('\n')
