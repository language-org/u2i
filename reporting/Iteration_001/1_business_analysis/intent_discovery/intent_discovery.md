file: intent_discovery.md   
topics: #nlp, #intent discovery  
workload: 1 day    

# Problems:

* There is huge benefits in being able to customize/augment intent analysis but Off-the-shelf NLU technologies (GAFAM etc) are opaque
* User inputs are ill-formed, disfluent, fragmentary, desultory, rambling (4) 
* To our knowledge, no existing free technology enables the (analysis?) of `intent expressiveness`, (comprehensive?) descriptive segmentation of intents   
* The current `dominant solutions` deployed in production uses `Semantic grammars` to detect intents but this approach is labor intensive and only applies to small well-mapped domains.
* [Q]: Can we produce meaningful labels from unlabelled data (i.e., similar to human-annotated labels)?

# Scope:

* Better Science on intent analyis 
* Technology watch on SOTA and challenges  
* Customizable artifact for intent analysis  
    * preprocessing  
    * discovery  
    * clustering
    * visualization  
        * different levels of expressiveness  
* Better skills and expertise  

# Impact:

    * Q&A ?
    * Summarization ?
    * Chatbots?
    * Text classification?

# Objectives:  

    * produce intents 
      * data structures that enable hierarchical level of expressivity (.JSON)
      * that are normative, exloitable, interpretable (e.g., JSON)
    * optional:
        * detect mood: 
            * e.g., declarative, imperative, interrogative ..., 
            * affect: positive/negative (anger, fear, ..)
            * language: slang, friendly, informal, formal  
        * confidence level
        * auto-detect failure situations (more teachable, debuggable):
          * out-of-scope
          * requires pragmatics (lack context)  
          * complex syntax (complex and/or compound sentences)
    * domain-specific

# Constrains:
    * ?

# Challenges

    * Intents are sometimes `implicit` (18)
        * Want to eat: e.g., "I'm so hungry" 

# Approaches
## Intent and entity discovery/parsing
* Approaches:
    * `Shallow semantics` (4,9) 
        * principle:
            * parsing tree (i.e., grammar) terminals substitution (4)
            * rules convert parsing tree (i.e., grammar) terminals to semantics  
        * hand-written CFG dictionary
            * pros:  
                * widely used  
            * cons:
                * can't deal w/ ambiguity
                * cost: labor intensive, manually constructed grammars 
        * technos:
            * `CSLU toolkit` (5, 6)
                * GUI
                * audience: 
                    * researchers        
    * `Semantic grammars` (4,9): 
        * principle:
            * Integrates syntactic parsing w/ semantic analysis                
                1. Syntactic structure is parsed by standard CFG parsing algo  
                1. Syntactic structure is converted to semantic structure by designating some non-terminal values as slots to be filled w/ corresponding terminals  
            * create frame-slot semantics : attribute value pairs (4) 
            * frame can be nested     
        * algos: 
            1. syntactic parsing
                * `CKY` (Cocke-Kasami-Younger) 
                    * techno:
                        * `nltk`:
                            * prototyping (1)
                * `Earley` parser  
                * `R.T.N`:  
                    * i.e., recursive transition networks  
            1. slot filling
                * Phoenix NLU (CU, CMU), vxml grammars             
        * pros:
            * widely used, almost all deployed spoken dialog systems (7)  
        * cons:
            * adaptation: not, domain & application-specific  
            * robustness: can't deal w/ ambiguity
                * solution: probabilistic semantic grammar (e.g., `TINA`), build rules manually then train probabilities w/data
            * cost: labor intensive, manually constructed semantic grammars  
            * maintainability: hard
        * use cases:
            * small well-mapped domains    
    * `Probabilistic semantic grammars` (4,9):   
        * principle:  
            * build grammars manually then train probabilities (machine learning) w/data  
        * pros:
            * robustness: improved compared to `Semantic grammars` and `Shallow semantics` 
            * ambiguity handling: yes
            * adaptation: yes, can be trained for new domains, apps  
        * cons:
            * cost: manually constructed grammar labor intensive  
        * use cases:
            * small well-mapped domains    
        * Models:  
            * `TINA` (4)  
    * `Probabilistic classification or sequence labelling` (9):
        * `Semantic H.M.M.`:    
            * i.e., semantic hidden markov model 
            * author: (Peraccini, 1991) (4,9,20)    
            * principle:
                * Hand-label a corpus w/ slot-names then learn word strings with slot-names w/ a HMM algo (9)
                * Bayesian modeling  
            * model:  
              * train:
                * input: hand-labelled corpus (word strings, slot-name)  
                * algo: `Viterbi` algorithm (an `HMM`)
              * predict:
                * input: utterance
                * output: slot-names                 
            * pros:
                * cost: none, learns slots so no hand written grammar 
                * ambiguity handling: can deal with ambiguity  
                * adaptation: yes, can be trained for new domains, apps  
            * cons:
              * interpretability?
        * `H.U.M.`:   
            * i.e., Hidden Understanding model (9)
            * principle: 
                * adds hierarchical structure to the `semantic H.M.M` (9)
                * combines `semantic grammar` and `semantic H.M.M approaches`
        * `MaxEnt-based` (4)         

    * Neural Nets (3)  
    * Other approaches: 
      * Two-stages:
        1. Intent detection: 
           * Present: in Questions which are assumed to typically contain intents (21)
           * Absent: in Answers which are assumed not to contain intents (21)
        2. Actionable intent parsing (verb-object):
           * techno:
             *  `Stanford CoreNLP dependency parser` (21)  

## Approach 1
    * Supervised intent classification  
        * model development and comparison on single label corpus
        * find explainable model with near SOTA performance

## Approach 2
    * Unsupervised intent (label) discovery  
        * clustering: flat and herarchical (2)
        * knowledge graph (semantic) based similarity metrics (my principal focus) 
        * intent discovery
            * Shallow semantics
            * Semantic grammars 
                * syntactic parsing
                * slot filling
            * Probabilistic semantic grammar
            * Semantic hidden markov models
            * Neural models (not a priority)  

* Corpus
    * Focus on Banking77 (financial)
    * Others identified (see my corpora survey)   


# References

[zotero_ok]
(1) https://www.ling.uni-potsdam.de/~scheffler/teaching/2016advancednlp/assignments/anlp16-assignment-3.pdf 
(2) https://github.com/Aadesh-Magare/PCFG_CKY_NLU/blob/master/Report.pdf   
(3) https://github.com/sz128/slot_filling_and_intent_detection_of_SLU  

[zotero_ko]
(4) http://courses.washington.edu/ling575/SPR2013/slides/ling575_class3_nlu.pdf  
(5) https://web.archive.org/web/20110817012415/http://www.cslu.ogi.edu/toolkit/  
(6) https://www.youtube.com/watch?v=ZrAlj7GQqjQ  
(7) http://courses.washington.edu/ling575/SPR2015/slides/ling575_class3_asr_nlu_vxml.pdf 
(8) http://cs.union.edu/~striegnk/courses/nlp-with-prolog/html/toc.html#label1  
(9) Jurafsky Book  

[zotero_ok]
(1) Gomaa, W. H., & Fahmy, A. A. (2013). A survey of text similarity approaches. International Journal of Computer Applications, 68(13), 13-18. (SEED)     
(2) Harispe, S., Ranwez, S., Janaqi, S., & Montmain, J. (2013). Semantic measures for the comparison of units of language, concepts or instances from text and knowledge base analysis. arXiv preprint arXiv:1310.1285.   
(3) Zhu, G., & Iglesias, C. A. (2016). Computing semantic similarity of concepts in knowledge graphs. IEEE Transactions on Knowledge and Data Engineering, 29(1), 72-85. (SEED)
(4) Sousa, R. T., Silva, S., & Pesquita, C. (2020). Evolving knowledge graph similarity for supervised learning in complex biomedical domains. BMC bioinformatics, 21(1), 6.  
(5) Maxat Kulmanov et al., “Semantic Similarity and Machine Learning with Ontologies,” Briefings in Bioinformatics, 2020.  
(6) Hatzivassiloglou, V., Klavans, J. L., & Eskin, E. (1999). Detecting text similarity over short passages: Exploring linguistic feature combinations via machine learning. In 1999 Joint SIGDAT conference on empirical methods in natural language processing and very large corpora.  
(7) Mihalcea, R., Corley, C., & Strapparava, C. (2006, July). Corpus-based and knowledge-based measures of text semantic similarity. In Aaai (Vol. 6, No. 2006, pp. 775-780).  
(8) Hatzivassiloglou, V., Klavans, J. L., & Eskin, E. (1999). Detecting text similarity over short passages: Exploring linguistic feature combinations via machine learning. In 1999 Joint SIGDAT conference on empirical methods in natural language processing and very large corpora.
(18) Yue Shang, Studies on User Intent Analysis and Mining (Drexel University, 2017).  
(19) https://www.mdpi.com/2071-1050/10/8/2731/htm  
(20) Pieraccini, R., & Levin, E. (1991). Stochastic representation of semantic structure for speech understanding. In Second European Conference on Speech Communication and Technology.  
(21) Nikhita Vedula et al., “Towards Open Intent Discovery for Conversational Text,” ArXiv:1904.08524 [Cs], April 17, 2019, http://arxiv.org/abs/1904.08524.  

[tosort]
https://fasttext.cc/docs/en/supervised-tutorial.html  
https://link.medium.com/3OQzNuKax7  
description: intent vs topic classification    
https://www.researchgate.net/profile/Amol_Ambardekar/publication/260662750_Context-Based_Bayesian_Intent_Recognition/links/546a82d60cf20dedafd389b7/Context-Based-Bayesian-Intent-Recognition.pdf  
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/06854367.pdf   
https://github.com/sebastianruder/NLP-progress/blob/master/english/intent_detection_slot_filling.md  