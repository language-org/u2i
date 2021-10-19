
file: intent_classification.md  
date: 2020_06_26.md

## Ideal requirements    

* To do: rank by business value  

## SOTA

  * Commercial APIs
	* `IBM Watson`
		* features
			* single intent classification
			* Entity classification
			* payment plan: commercial
			* most popular
		* limitations
			* no multi-intent classification (severe issue)
			* payment plan: commercial
			* No info on whether joint intent-Entity classification  
		* training input:
		* format:
			* CSV
		* content:
			* args:
				* utterances 
				* intent labels
				* entities w/ values, no annotation needed
		* Output:
			* format:
				* JSON
			* content:
				* intent w/ confidence
				* entities w/ confidence		  
	* `Google dialogflow`
		* features
			* intent classification
			* entity classification
			* payment plan: commercial
			* most popular			  
		* limitations
		* no multi-intent classification (severe issue)				
		* No info on whether joint intent-Entity classification  
		* training input:
		* format:
			* JSON
		* content:
			* args:
				* utterances		  			
				* annotated intents 
				* annotated entities
				* values/synonyms
				* expected output entities for each intents  
		* Output:
			* format:
				* JSON
			* content:
				* intent
				* entities
				* An overall score  
	* `Mircrosoft LUIS`
		* features
			* intent classification
			* Entity classification
			* payment plan: commercial
			* most popular			  
		* limitations
			* no multi-intent classification (severe issue)
			* No info on whether joint intent-Entity classification  
			* Max training set at 10K
		* training input:
			* format:
				* JSON
			* content:
				* phrase list
				* regexp patterns as model features  
				* hierarchical and composite entities  
				* expected entities for each intent  
		* Output:
			* format:
				* JSON
			* content:
				* intent w/ confidence
				* entities with scores
	* `Facebook Wit.ai`
		* features 
			* intent classification
			* Entity classification	  
			* payment plan: commercial
			* most popular			  
		* limitations
	* `Amazon lex`
		* features
			* intent classification
			* Entity classification
		*
	* `Recast.ai`
		* features
			* intent classification
			* Entity classification
		*
	* `Botfuel.io`
		* features
				* intent classification
				* Entity classification	  
	* `Snips.ai` (6)
		* features
    		* intent classification
    		* Entity classification
    		* Open source
  		* training input:
    		* .YAML dictionary of intents, sample utterances + more info
  		* input
    		* an utterance  
  		* output:
    		* .JSON intent and entities  
    	* pros:
        	* quick to learn
        	* easy to train
	* `RASA`
		* features
			* open source
			* most popular			  
		* limitations
			* no multi-intent classification (big issue)
			* no joint intent-Entity classification  
			* not dialog context sensitive
		* training input:  
		* input:  
    		* an utterance  
		* format:
			* JSON, Markdown
		* content:
			* args:
				* Utterances 
				* annotated intents
				* annotated entities
			* optional: synonyms, regexp  
		* predicted Output:
			* format:
				* .JSON
			* content:
				* intent
				* intent ranking w/ confidence
				* entities w/o scores
    * `Mycroft`
      * `padatious` (4)  
        * principle: neural network
        * training input:	
          * file named w/ intent and > utterance samples
        * pros:  
          * lightweight, quick to train on simple devices (e.g., embedded)  
        * cons:
          * opaque  
      * `Adapt` (5)  
        * principle: keyword based
        * pros
          * lightweight, simplest
          * interpretable
        * cons:
          * labor intensive  
	* Intent classification Benchmarking
		* Corpus: `Liu 2019` [1]
			* precision:	
				* all: ~85%
				* Watson >> Dialogflow, LUIS, Rasa
			* recall: 
				* all: ~85% 
			* F1:
				* all: ~85%		
		* Corpus: `Clinc` [3]
			* Accuracy:   
				* DialogFlow, Rasa: ~91%  
	* NER Benchmarking
		* precision	:   	  
			* Dialogflow, LUIS, RASA: ~80%
			* Watson: 35%
		* recall
			* ~ 70%
		* F1
			* ~ 75% (except Watson: 50%)
  * Academic models 
	* Intent classif. benchmark
		* Corpus: `Clinc` [3]
			* Accuracy:
				* BERT: 96% (in-scope queries [3])
				* BOW+SVM: 91% (in-scope queries [3])
	* NER Benchmark
		* Corpus: `Newswire articles` [2]
			* Take home message: 
				* NN > classical ML by 0.36% in English
			* Performance: 			
				* F1
					* Classical ML: Agerri & Rigau (2016): 91.36% 
					* DL: Chiu & Nichols (2015): 91.62%
						
  * Questions: 
	* We can upload our own corpus (text / intent) to train the model 
	* Can we download the trained model to use for a client project?  		
	* is the model lighweight and heavy (GB)  
	* Does the user obtain their own version of the model in this case?  
	* Multilabel intent classification  		

## References  

[1] Liu, X., Eshghi, A., Swietojanski, P., & Rieser, V. (2019). Benchmarking natural language understanding services for building conversational agents. arXiv preprint arXiv:1903.05566.   
[2] Vikas Yadav and Steven Bethard, “A Survey on Recent Advances in Named Entity Recognition from Deep Learning Models,” ArXiv Preprint ArXiv:1910.11470, 2019.    
[3] Stefan Larson et al., “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (EMNLP-IJCNLP 2019, Hong Kong, China: Association for Computational Linguistics, 2019), 1311–1316, https://doi.org/10.18653/v1/D19-1131.  
(4) https://mycroft-ai.gitbook.io/docs/skill-development/user-interaction/intents/padatious-intents  
(5) https://mycroft-ai.gitbook.io/docs/mycroft-technologies/adapt  
(6) https://github.com/snipsco/snips-nlu/blob/master/docs/source/tutorial.rst  