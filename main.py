""" 
Ask Vachanamrut 
author: ytpatel3 
""" 

import re 
# pip install rake-nltk 
from rake_nltk import Rake 
# pip install 'aisuite[openai]' 
import aisuite as ai 

def main(): 
    print("\nThe Vachanamrut —the paramount scripture of the BAPS Swaminarayan faith— is composed of 273 spiritual discourses delivered by Bhagwan Swaminarayan from 1819 to 1829 that introduces the novel philosophy of Akshar-Purushottam Darshan. It is filled with sound logic, illuminating metaphors and analogies, and divine revelations that provide philosophical and practical answers to questions in all aspects of one’s life.")
    user_input = input("\nWhat spiritual/philosophical question would you like the answer to?\n")
    
    # use regex to remove punctuation and then split words
    text = re.sub(r"[^a-zA-Z\s]","",user_input)
    
    # use Rake-NLTK to get important words/phrases from user's input
    rake_nltk_var = Rake()
    rake_nltk_var.extract_keywords_from_text(text)
    keywords = rake_nltk_var.get_ranked_phrases()

    # Sample prompt: I am feeling lonely and unmotivated. I want inspiration and courage.
    user_feelings = "".join(keywords)
    print("\n" + user_feelings + "\n")

    
if __name__ == "__main__": 
    main()