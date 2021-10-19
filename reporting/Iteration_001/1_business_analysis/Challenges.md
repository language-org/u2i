
file: Challenges.md  

* `Multi-intent classification` [3,5]
    * rationale: 
        * very common in natural dialogue (but rarely > 2 intents[4]), more expressive    
        * smoother conversation than expressing each intent separately [4]       
        * single query - independent intents 
            * e.g., 'dim the lights, cut the music and play Black Mirror on tv'
        * single query - dependent intent
            * e.g., 'turn the lights off in the living room, dining room, kitchen and bedroom'
    * Current SOTA  
        * Predominant focus on `Single-intent classification` systems [4] 
            * cons: 
            * forces unnatural conversation [4]
            * shows slower processing as requires more turns [4]
                * e.g., one simple turn:
                    * turn 1: 'dim the lights, cut the music and play Black Mirror on tv' 
                    --> dim_light, cut_music, play_black_mirror 
                * has to be expressed by 3 turns:  
                    * turn 1: 'dim the lights' 
                    --> dim_light 
                    * turn 2: 'cut the music' 
                    --> cut_music
                    * turn 3: 'play Black Mirror on tv' 
                    --> play_black_Mirror 
            * Low expressivity: 
                * The classified intent is less informative than human-level recognition
                * e.g., Instead of --> card_arrival, 
                    * "I ordered my card but it still isn't here. When will it arrive" --> card_arrival, when 
                    * "I ordered my card but it still isn't here. Who should I talk to" --> card_arrival, contact
    * Approaches: 
      * `Sentence segmentation` (e.g., BIO chunking + intent classification for each chunk)
        * cons: 
          * ignores shared information between chunks [4]  
            * e.g., 1:[find avatar] and 2:[play it] --> 2 is uninformative w/o 1
* `Joint Intent-Entity classification` 
    * rationale: entities depend on intents  
* `Context dependence`
    * intent & entity recognition dependent on the context
* `Cross-domains` or high performance for a specific industry  
* `Out-of-scope fall-back response` [1]  
    * rationale: 
        * being able to say: "Out of my scope" when a query's intent is not known --> fallback response  
        * prevent doing the wrong thing  
    * techniques:
      * Clustering-based (8)
* `Robustness to jargon`:
    * rationale: professional settings use lots of technical jargon  
* `Data sparsity`     
    * rationale: dataset with labels are often small in the commercial context  
    * performance should be robust to decreasing sample size per intent 
        * approaches
            * `few shot learning`
            * `Shannon-entropy`: Word representation by entropy distribution [6]  
                * pros: 
                    * very simple
                    * no representation training like word2vec  
                * cons: 
                    * low intepretability: requires PCA  
* `Interpretable representation`
    * Current SOTA
        * Opaque Neural networks 
    * Approaches
        * Interpretable word embeddings via informative priors [7]  
* `Ambiguous semantics`:
    * can't always rely on word semantics to capture intent (9)
    * fine-grained discrimination might be needed (9)
    * e.g., 
      * Card Lost: "Could you assist me in finding my lost card?"
      * Link to Existing Card: "I found my lost card. Am I still able to use it?"
      * Reverted top-up: "Hey, I thought my topup was all done but now the money is gone again – what’s up with that?"
      * Failed Top-up: "Tell me why my topup wouldn’t go through?"
* `Implicit vs. explicit intent`


## References  

[1] Stefan Larson et al., “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (EMNLP-IJCNLP 2019, Hong Kong, China: Association for Computational Linguistics, 2019), 1311–1316, https://doi.org/10.18653/v1/D19-1131. 
[3] https://stackoverflow.com/questions/56064650/multi-intent-natural-language-processing-and-classification   
[4] Puyang Xu and Ruhi Sarikaya, “Exploiting Shared Information for Multi-Intent Natural Language Sentence Classification.,” in Interspeech, 2013, 3785–3789.   
[5] Grigorios Tsoumakas and Ioannis Katakis, “Multi-Label Classification: An Overview,” International Journal of Data Warehousing and Mining (IJDWM) 3, no. 3 (2007): 1–13.    
Info: +2000 citations    
[6] Perevalov, A., Kurushin, D., Faizrakhmanov, R., & Khabibrakhmanova, F. (2019). Question Embeddings Based on Shannon Entropy: Solving intent classification task in goal-oriented dialogue system. arXiv preprint arXiv:1904.00785.  
[7] Bodell, M. H., Arvidsson, M., & Magnusson, M. (2019). Interpretable Word Embeddings via Informative Priors. arXiv preprint arXiv:1909.01459.  
(8) https://dke.maastrichtuniversity.nl/jan.niehues/wp-content/uploads/2020/09/MASTER-THESIS-Camiel-Kerkhofs.pdf   
(9) Iñigo Casanueva et al., “Efficient Intent Detection with Dual Sentence Encoders,” ArXiv Preprint ArXiv:2003.04807, 2020.  