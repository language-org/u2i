
file: intent_modeling.md  
topics: #nlp, #intent granularity, #intent expressivity, #intent definition    

# Definition frameworks

  ## Hollerit et al, 2013

  * definition: 
    * an intent utterance contains at least one verb and describes the user’s intention to commit an activity in a recognizable way
  * `explicit` vs `implicit` 
  
  ## Wang et al., 2015

  * definition: 
    * as Hollerit et al, 2013  
  * `intent-indicator`: group of terms used to express intents: verb or infinitive phrase that immediately follows a subject word   
    * e.g., In “I want to buy an xbox”: intent-indicator = “want to”  
  * `Intent-Keyword`: noun, verb, multi-word verb or compound noun contained in a verb or noun phrase which immediately follows an intent-indicator, 
    * e.g., in “buy an xbox” intent-keywords = (“buy”, “xbox”)
  
  ## Vedula et al., 2019

    * Explicit intents are actionable (9,10,11)  
      * 2 parts: 
        * an `action` and an `object` (9,10,11)  

  ## IBM Watson

  * e.g.,:    
    * #Covid_Am_I_Infected (6)
    * #weather_conditions
    * #pay_bill
    * #excalate_to_agent  
  
  ## Google
  
  * e.g.,:
    * (14)  

  ## RASA

  * e.g., (light, one word): 
    * Intent:inform 
      * Utterance: "I like puppies" 
      * "Steeve, we'll leave at 4"
    
  ## Mycroft AI adapt

  * e.g., of a single intent (rich data structure):
    * Utterance: "Play Mozart in Spotify"  
    * IntentType: #MusicType
    * MusicVerb: "play"
    * MusicKeyWord: "spotify"  
    * Artist: "Mozart"

  ## ATIS corpus
  
  * e.g.,:
    * Utterance: What flights are available from Pittsburgh to Baltimore on thursday morning
    * intent: #flight_info (8)

  ## Data-driven observations

  * intent:
    * syntax: 
      * `verb phrases` (VP) or `predicate`: 
        * `Verb - Noun Phrase` subcategory: "search the truth" (p400 Jurafsky)
        * `transitive verbs`, not intransitive
        * technique:
          * `constituency parsing`
      * optionally: 
        * `intent-indicator`: "want to" + "search the truth"
    * tense: 
      * `present`: not "searched truth"  
  * Method:
    * exploring Banking77 dataset
    * looking attention (13)

# Intent taxonomy

  ## Classes
  ### Broder, 2002, in information retrieval    
  
  * navigational  
  * transactional  
  * informational  
  * see hierarchy of intents (4, p.9)  
  * see `speech acts`:
    * requests, commands, ...
  
  ## Dimensionality ("facets")
  ### Calder ́on-Benavides, 2010 in information retrieval    

   * Genre, objective, specificity, scope, topic, task, authority sensitivity, spatial sensitivity and time sensitivity

  ## Models

  * In the health community, Cai et al [3] used hierarchical clustering to learn a taxonomy of intent classes (13)  

# Intent abstraction

  * Conceptual hierarchy:  
    * [sl] e.g., 
      * (intent:#inform, (taxonomy:superclass, expressivity:low))
        * --> (intent:#departure_time, (taxonomy:class, expressivity:high))  
          * --> (entities:#receiver: steeve, (taxonomy:class, expressivity:high))   
      * note: #departure_time could be called an entity also (e.g., slot_filling)  

# Modeling

* find a normative model of intent query (12)
  * e.g., create a normative model of an intent query
    * focus on the norm and drop outlier syntaxes (noise)(12)

# References  

(1) Yue Shang, Studies on User Intent Analysis and Mining (Drexel University, 2017).    
(2) Bernd Hollerit, Mark Kr ̈oll, and Markus Strohmaier. Towards linking buyers and sellers: detecting commercial intent on twitter. In Proceedings of the 22nd International Conference on World Wide Web, pages 629–632. ACM, 2013.  
(3) Jinpeng Wang, Gao Cong, Wayne Xin Zhao, and Xiaoming Li. Mining user intents in twitter: A semi-supervised approach to inferring intent categories for tweets. In AAAI, pages 318–324, 2015.    
(4) Andrei Broder. A taxonomy of web search. In ACM Sigir forum, volume 36, pages 3–10. ACM, 2002.  
(5) Liliana Calder ́on-Benavides, Cristina Gonz ́alez-Caro, and Ricardo Baeza-Yates. Towards a deeper understanding of the users query intent. In SIGIR 2010 Workshop on Query Repre- sentation and Understanding, pages 21–24, 2010.
(6) https://cloud.ibm.com/docs/assistant?topic=assistant-intents  
(7) https://www.youtube.com/watch?v=JOf6CNJUzEo  
(8) https://github.com/sebastianruder/NLP-progress/blob/master/english/intent_detection_slot_filling.md  
(9) Nikhita Vedula et al., “Towards Open Intent Discovery for Conversational Text,” ArXiv:1904.08524 [Cs], April 17, 2019, http://arxiv.org/abs/1904.08524.  
(10) Huadong Chen, Shujian Huang, David Chiang, and Jiajun Chen. 2017. Improved neural machine translation with a syntax-aware encoder and decoder. arXiv preprint arXiv:1707.05436 (2017).
(11) Jinpeng Wang, Gao Cong, Wayne Xin Zhao, and Xiaoming Li. 2015. Mining User Intents in Twitter: A Semi-Supervised Approach to Inferring Intent Categories for Tweets.. In AAAI.   
(12) Bimal Viswanath et al., “Towards Detecting Anomalous User Behavior in Online Social Networks,” in 23rd ${$USENIX$}$ Security Symposium (${$USENIX$}$ Security 14), 2014, 223–238.  
(13) Nikhita Vedula et al., “Towards Open Intent Discovery for Conversational Text,” ArXiv:1904.08524 [Cs], April 17, 2019, http://arxiv.org/abs/1904.08524.  
(14) https://www.youtube.com/watch?v=Ov3CDTxZRQc  